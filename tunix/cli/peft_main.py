"""Main entry point for PEFT training."""
from collections.abc import Callable
from typing import Any
from absl import app
from flax import nnx
import jax
from tunix.cli import config
from tunix.cli.utils import model as model_lib
from tunix.cli.loss import *
from tunix.examples.data import ift_dataset as data_lib_ift
from tunix.sft import subset_trainer
from tunix.sft import utils
import optax
from typing import Any, Callable
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from functools import partial
from flax.core import FrozenDict
from tunix.sft import subsel_utils



def build_optimizer(cfg):
    # --- Scheduler ---
    '''
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    '''

    def linear_warmup_decay_schedule(warmup_steps, total_steps, peak_lr, end_lr=0.0):
      warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
      decay = optax.linear_schedule(peak_lr, end_lr, total_steps - warmup_steps)
      return optax.join_schedules([warmup, decay], [warmup_steps])

    # lr = 2e-5
    lr = cfg["optimizer_config"]["learning_rate"]
    warmup_ratio = cfg["optimizer_config"]["warmup_ratio"]
    print("$$$$$$$$$$$$$$$$$$$")
    print("LEARNING RATE", lr)
    print("$$$$$$$$$$$$$$$$$$$")

    total_steps = cfg["training_config"]["max_steps"]

    warmup_steps = int(warmup_ratio*total_steps)
    # warmup_steps = 700

    decay_steps = total_steps - warmup_steps
    weight_decay = 0.0
    grad_norm = 1.0

    # schedule = optax.warmup_cosine=_decay_schedule(
    #   init_value=0.0,
    #   peak_value=lr,
    #   warmup_steps=warmup_steps,
    #   decay_steps=decay_steps,
    #   end_value=0.0,
    # )

    schedule = linear_warmup_decay_schedule(
      warmup_steps=warmup_steps,
      total_steps=total_steps,
      peak_lr=lr,
      end_lr=0.0
    )
    optim = cfg["optimizer_config"]["opt_type"]
    print("$$$$$$$$$$$$$$$$$$ OPTIM", optim)
    
    if optim == "sgd":
      tx = optax.sgd(learning_rate=schedule)
    if optim == "muon":
      tx = optax.contrib.muon(learning_rate=schedule, weight_decay=weight_decay)
    if optim == "adamw":
      # tx = optax.schedules.inject_hyperparams(optax.adamw)(learning_rate=schedule, weight_decay=weight_decay)
      tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

    tx = optax.chain(
      optax.clip_by_global_norm(grad_norm),
      tx,
    )

    return tx, schedule


def prepare_input(tokens, pad_id, is_causal=True):
  pad_mask = jax.numpy.ones_like(tokens)

  positions = utils.build_positions_from_mask(pad_mask)
  if is_causal:
    attention_mask = utils.make_causal_attn_mask(pad_mask)
  else:
    attention_mask = utils.make_self_attn_mask(pad_mask)

  return {
    "input_tokens": tokens,
    "positions": positions,
    "attention_mask": attention_mask,
  }

def gen_model_input_fn_ift(x: subset_trainer.TrainingInput, pad_id, is_causal):
  ret = prepare_input(x["input_tokens"], pad_id, is_causal)
  ret["input_mask"] = x["input_mask"]
  if "meta" in x.keys():
    ret["meta"] = x["meta"]
  return ret


class PeftPipeline(config.HyperParameters):

  def run_peft_trainer(self):
    """Run the PEFT trainer."""
    mesh: jax.sharding.Mesh = self.create_mesh('model_config')
    model: nnx.Module | None = None
    tokenizer: Any | None = None
    my_gen_model_input_fn: (
        Callable[[subset_trainer.TrainingInput], dict[str, Any]] | None
    ) = None
    from omegaconf import OmegaConf
    # self.config = OmegaConf.to_object(self.config)
    self.config["task_config"]["config"] = OmegaConf.load(self.config["task_config"]["config"])
    print("Loaded task config", self.config["task_config"]["config"])
    print("###########")
    print("###########")
    print("###########")
    print("###########")
    
    model, tokenizer_path = model_lib.create_model(
        self.config['model_config'], self.config['tokenizer_config'], mesh
    )

    if model is None:
      raise ValueError('model is None')
    tokenizer = model_lib.create_tokenizer(
        self.config['tokenizer_config'], tokenizer_path
    )
    
    steps = self.config["training_config"]["max_steps"]
    

    print("EOS", tokenizer.eos_id())
    print("PAD", tokenizer.pad_id())
    # assert tokenizer.pad_id() == tokenizer.eos_id()


    optimizer, schedule = build_optimizer(self.config)
    optimizer_last, schedule1 = build_optimizer(self.config)
    
    
    task_config = self.config["task_config"]["config"]
    print("task config", task_config)
    PAD_ID = tokenizer.pad_id()
    EOS_ID = tokenizer.eos_id()
    TASK = self.config["task_config"]["task"]

    if TASK == "ift":
      my_gen_model_input_fn = partial(gen_model_input_fn_ift, 
                                      pad_id=PAD_ID, is_causal=True)
      project = None
      
      my_datalib = data_lib_ift
      train_loss = loss_unpacked
      eval_loss = default_loss_fn
      dataset_name=self.config['dataset_name']
      eval_fn = None
      cache_dir=None

    subsel = self.config["subset_select"]["enabled"]
    buffer = 1
    if subsel:
      buffer = self.config["subset_select"]["buffer"]
    
    train_ds, eval_ds, dev_ds, data_meta = my_datalib.create_datasets(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        global_batch_size=self.config['batch_size']*buffer,
        eval_global_batch_size=self.config['eval_batch_size'],
        max_target_length=self.config['max_target_length'],
        num_train_epochs=100, #FIXME: poor mans infinity
        tokenizer=tokenizer,
        split_ratio=self.config['eval_split'],
        config=self.config,
        subsel_bs = self.config["batch_size"]*get_train(self.config)
    )
    # ds = train_ds._data_source
    ds = train_ds
    def get_len(ds):
      if ds is None: return None
      try: l = len(ds._data_source)
      except: 
        try: l = len(ds.dataset)
        except: l = len(ds)
      return l
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("Len Dataset(Million):", get_len(ds)*self.config['batch_size']/1e6)
    print("Num Batches:", get_len(ds))
    print("Len Eval Dataset:", get_len(eval_ds))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # tx = optax.MultiSteps(
    #       optimizer, 1
    #   )
    # tx  = nnx.Optimizer(model, tx, wrt=nnx.LoRAParam)
    # jax.debug.print("opt {}", tx.opt_state.inner_opt_state[1].hyperparams)

    
    trainer = subset_trainer.PeftTrainer(
      model,
      optimizer,
      subset_trainer.TrainingConfig(
        **self.obtain_training_config_dict('training_config')
      ),
      optimizer_last=optimizer_last,
      has_aux=True,
      schedule=schedule,
      config=FrozenDict(self.config),
      train_loss=train_loss,
      eval_loss=eval_loss,
      eval_fn=eval_fn,
      project=project
    )
    
    trainer = trainer.with_gen_model_input_fn(my_gen_model_input_fn)
    # jax.set_mesh(mesh)
    with mesh:
      trainer.train(train_ds, data_meta["num_train_sources"], eval_ds, dev_ds=dev_ds)

def get_train(config):
  if not config["subset_select"]["enabled"]: return 1
  _ratio = config["subset_select"]["ratio"]
  if config["subset_select"]["mode"] == "full": _ratio = 1
  _buffer = config["subset_select"]["buffer"]
  batch_to_buffer =int(_buffer*_ratio) 
  return batch_to_buffer


def main(argv, **kwargs):
  pipeline = PeftPipeline(argv, **kwargs)
  pipeline.run_peft_trainer()


if __name__ == '__main__':
  app.run(main)