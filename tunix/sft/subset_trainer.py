from collections.abc import Iterable
import contextlib
import dataclasses
import time
from typing import Any, Callable, Dict, List, Tuple

from absl import logging
import flax
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import jax.sharding as shd
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.sft import checkpoint_manager
from tunix.sft import inflight_throttler
from tunix.sft import metrics_logger
from tunix.sft import profiler
from tunix.sft import progress_bar
from tunix.sft import sharding_utils
from tunix.sft import utils
from functools import partial
from tunix.sft import subsel_utils
from tunix.sft.subsel.grads import grad_filter
@flax.struct.dataclass(frozen=True)
class TrainingConfig:
  """Configuration for the trainer."""

  eval_every_n_steps: int
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None

  # If set, the checkpoints will be saved to this path. Checkpoints
  # contains the model params and the train data iterator state.
  checkpoint_root_directory: str | None = None
  # Checkpoint configurations. If None, the default options will be used.
  checkpointing_options: ocp.CheckpointManagerOptions | None = None

  # Configs for the metrics logger.
  metrics_logging_options: metrics_logger.MetricsLoggerOptions | None = None

  # Configs for the profiler.
  profiler_options: profiler.ProfilerOptions | None = None

  data_sharding_axis: Tuple[str, ...] = ("fsdp",)

  # Controls how many train_steps can be scheduled ahead of time.
  max_inflight_computations: int = 2

  # Prefix for metric names for logging. Not sticking it in
  # `metrics_logging_options` because the latter is optional.
  metric_prefix: str = ""

  # Progress bar description.
  pbar_description: str | None = "Training"

  def get_with_default(self, key: str, default: Any) -> Any:
    val = getattr(self, key)
    if val is None:
      return default
    return val


@flax.struct.dataclass(frozen=True)
class TrainingInput:
  # Input tokens provided to the model.
  input_tokens: jax.Array | np.ndarray

  # A mask that determines which input tokens are valid.
  input_mask: jax.Array | np.ndarray


@dataclasses.dataclass(slots=True, kw_only=True)
class MetricsBuffer:
  """Metrics collected for a specific step.

  Attributes:
    step: The training step number.
    losses: A list of loss values recorded within this step (e.g., across
      gradient accumulation steps).
    step_time_deltas: A list of time deltas for each computation within this
      step.
    additional_metrics: Dictionary for storing additional metrics. The key is
      the metric name, and the value is a tuple containing a list of metric
      values and a callable to aggregate them.
  """

  step: int
  losses: List[ArrayLike]
  step_time_deltas: List[float]
  additional_metrics: Dict[str, ArrayLike] = dataclasses.field(default_factory=dict)

  @property
  def loss(self):
    """Returns the mean of the recorded losses for the step."""
    return np.mean(self.losses)

  @property
  def step_time_delta(self):
    """Returns the mean of the recorded step time deltas for the step."""
    return np.mean(self.step_time_deltas)


class PeftTrainer:
  """PEFT trainer for LoRA. Only LoRA parameters are updated.

  Attributes:
    model: The model to train.
    config: The training config.
    optimizer: The optimizer to use. To monitor the learning rate at each step,
      use `optax.schedules.inject_hyperparams` to inject learning rate as a
      hyperparameter. For example: ``optimizer =
      optax.schedules.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule)``
    loss_fn: The loss function to use.
    eval_loss_fn: The loss function to use for evaluation.
    gen_model_input_fn: The function to generate model input from training
      input.
    checkpoint_manager: The checkpoint manager to use.
    metrics_logger: The metrics logger to use.
    is_managed_externally: Whether the trainer is managed externally.
  """

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
      optimizer_last: optax.GradientTransformation,
      train_loss: Callable,
      eval_loss: Callable ,
      eval_fn: Callable,
      has_aux: bool,
      schedule,
      config,
      project=None
  ):
    self._validate_config(training_config)
    self.model = model
    self.config = training_config
    self.fullConfig = config
    self.schedule = schedule
    self._lora_enabled = utils.is_lora_enabled(self.model)
    if training_config.gradient_accumulation_steps is not None:
      optimizer = optax.MultiSteps(
          optimizer, training_config.gradient_accumulation_steps
      )
      optimizer_last = optax.MultiSteps(
          optimizer_last, training_config.gradient_accumulation_steps
      )
    if self._lora_enabled:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.LoRAParam)
      self.optimizer_last = nnx.Optimizer(self.model, optimizer_last, wrt=partial(grad_filter, [27], True))
    else:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.Param)
    self.project = project
    self.optimizer_head = None
    self.loss_fn = train_loss
    self.eval_loss_fn = eval_loss
    self.gen_model_input_fn = lambda x: x
    self.checkpoint_manager = checkpoint_manager.CheckpointManager(
        root_directory=self.config.checkpoint_root_directory,
        options=self.config.checkpointing_options,
    )
    self.metrics_logger = metrics_logger.MetricsLogger(
        self.config.metrics_logging_options,
        metric_prefix=self.config.metric_prefix,
    )
    self.is_managed_externally = False

    self._train_steps = 0  # represent # of times model has been updated
    self._iter_steps = 0  # represent # of times trainer has looped
    self._throttler = inflight_throttler.InflightThrottler(
        max_inflight=training_config.max_inflight_computations
    )
    self._mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN
    self._has_aux = has_aux
    self._pbar = None

    self._iter_steps = self._train_steps * self.config.get_with_default(
        "gradient_accumulation_steps", 1
    )

    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None
    self._prof = profiler.Profiler(
        initial_step=self._iter_steps,
        max_step=self.config.max_steps,
        profiler_options=self.config.profiler_options,
    )
    self._buffered_train_metrics: MetricsBuffer | None = None
    self._prev_buffered_train_metrics: MetricsBuffer | None = None
    self._buffered_eval_metrics: MetricsBuffer | None = None
    self.rng = jax.random.key(0)
    self.eval_fn = eval_fn
    self._grad_method = self.fullConfig["task_config"]["config"]["train_grad_method"]
    self.eval_ds_back = None
    self.moments = None
    self.meta = None
    self.moments_notset = True
    self.cache_train = None
    self.cache_val = None

  def _validate_config(self, training_config: TrainingConfig):
    if (
        training_config.gradient_accumulation_steps is not None
        and training_config.eval_every_n_steps
        % training_config.gradient_accumulation_steps
        != 0
    ):
      raise ValueError(
          "eval_every_n_steps must be divisible by gradient_accumulation_steps,"
          f" but got {training_config.eval_every_n_steps} and"
          f" {training_config.gradient_accumulation_steps}"
      )

  def clear_jit_cache(self):
    """Clears the JIT cache of the train and eval step functions.

    This function should be called when the trainer is being reused after
    overiding the training related states, for example, the loss function.
    """
    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None

  def with_gen_model_input_fn(self, gen_model_input_fn: Callable):
    self.clear_jit_cache()
    self.gen_model_input_fn = gen_model_input_fn
    return self

  def _compute_grads_actual(
      self, model: nnx.Module, inputs: Any, optimizer_head, project, meta
  ):
    """Compute gradients via standard backpropagation."""
    step, loss_mask = meta["step"], meta["loss_mask"]
    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )
    out, grads = grad_fn(model, inputs, {"step": step, "loss_mask": loss_mask})
    return out, grads

  def _compute_grads(
      self, model: nnx.Module, inputs: Any, optimizer_head, project, meta
  ):
    """Dispatch gradient computation based on config.

    Supported methods:
      - "actual" (default): standard backpropagation via nnx.value_and_grad.

    New methods can be added by implementing a _compute_grads_<name> method
    and referencing it in the config under
    task_config.config.train_grad_method.
    """
    method = self._grad_method
    compute_fn = getattr(self, f"_compute_grads_{method}", None)
    if compute_fn is None:
      raise ValueError(
          f"Unknown train gradient method '{method}'. "
          f"Implement '_compute_grads_{method}' or use one of: actual."
      )
    return compute_fn(model, inputs, optimizer_head, project, meta)

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, optimizer_head, project, inputs: Any, meta
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    out, grads = self._compute_grads(model, inputs, optimizer_head, project, meta)
    optimizer.update(model, grads)
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None


  def _eval_step(
      self, model: nnx.Module, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    inputs = self.gen_model_input_fn(inputs)
    model.eval()
    out = self.eval_loss_fn(model, inputs)
    model.train()
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None

  def _shard_optimizer(self, mesh: shd.Mesh, optimizer) -> None:
    if mesh.empty:
      return
    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)

    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, optimizer_pspecs
    )
    nnx.update(optimizer, optimizer_sharded_state)

  def jit_train_and_eval_step(self, skip_jit: bool = False):
    if skip_jit:
      return self._train_step, self._eval_step
    if self._jitted_train_step_fn is None:
      mesh = pxla.thread_resources.env.physical_mesh
      self._shard_optimizer(mesh, self.optimizer)
      if self.optimizer_head is not None:
        self._shard_optimizer(mesh, self.optimizer_head)
      self._jitted_train_step_fn = nnx.jit(
          self._train_step, donate_argnames=("model", "project", "optimizer", "optimizer_head")
      )
      self._jitted_eval_step_fn = nnx.jit(
          self._eval_step, donate_argnames=("model",)
      )
    return self._jitted_train_step_fn, self._jitted_eval_step_fn

  def _shard_input(self, input_data):
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty:
      return input_data

    # Check if the input is already sharded with the target mesh to avoid
    # re-sharding.
    is_sharded = jax.tree.map(
        lambda x: isinstance(x, jax.Array)
        and hasattr(x, "sharding")
        and hasattr(x.sharding, "mesh")
        and x.sharding.mesh == mesh,
        input_data,
    )
    if all(jax.tree.leaves(is_sharded)):
      return input_data

    pspec = shd.PartitionSpec(*self.config.data_sharding_axis)

    with jax.transfer_guard("allow"):
      return jax.tree.map(
          lambda x: jax.make_array_from_process_local_data(
              sharding_utils.get_sharding(x, mesh=mesh, pspec=pspec), x
          ),
          input_data,
      )

  def _try_get_learning_rate(self) -> float | None:
    """Returns the learning rate from the optimizer state if available."""
    try:
      return self.optimizer.opt_state.inner_opt_state[1].hyperparams["learning_rate"].value
    except AttributeError:
      for chainpart in self.optimizer.opt_state:
        if isinstance(chainpart, optax.EmptyState):
          break
        if hasattr(chainpart, "hyperparams"):
          return chainpart.hyperparams["learning_rate"].value
      return None

  def _log_metrics(
      self,
      loss: ArrayLike = None,
      step: int | None = None,
      step_time_delta: float | None = None,
      additional_metrics: Dict[str, ArrayLike] | None = None,
  ):
    """Logs the metrics to the metrics logger and console."""
    if loss is not None:
      perplexity = np.exp(loss)
      self.metrics_logger.log("loss", loss, self._mode, step)
      self.metrics_logger.log("perplexity", perplexity, self._mode, step)
    learning_rate = self._try_get_learning_rate()
    if learning_rate is not None:
      self.metrics_logger.log(
          "learning_rate", jax.device_get(learning_rate), self._mode, step
      )
    if step_time_delta is not None:
      self.metrics_logger.log(
          "step_time_sec", step_time_delta, self._mode, step
      )
      self.metrics_logger.log(
          "steps_per_sec", 1.0 / (step_time_delta + 1e-9), self._mode, step
      )

    if self._mode == metrics_logger.Mode.TRAIN:
      logging.info(
          "Train step %d training loss: %f",
          step,
          loss,
          # perplexity,
      )
    for k, v in (additional_metrics or {}).items():
      self.metrics_logger.log(k, v, self._mode, step)

  def _buffer_metrics(
      self,
      metrics_buffer: MetricsBuffer | None,
      loss: ArrayLike,
      step: int,
      step_time_delta: float = 0.0,
      additional_metrics: Dict[str, ArrayLike] = {},
  ) -> MetricsBuffer:
    """Buffers metrics for the current step."""
    loss = np.array(loss)
    if metrics_buffer is None:
      metrics_buffer = MetricsBuffer(
          step=step,
          losses=[loss],
          step_time_deltas=[step_time_delta],
      )
      if additional_metrics:
        for name, val in additional_metrics.items():
          metrics_buffer.additional_metrics[name] = [np.array(val)]
    else:
      assert metrics_buffer.step == step
      metrics_buffer.losses.append(loss)
      metrics_buffer.step_time_deltas.append(step_time_delta or 0.0)
      if additional_metrics:
        for name, val in additional_metrics.items():
          if name not in metrics_buffer.additional_metrics:
            metrics_buffer.additional_metrics[name] = [np.array(val)]
          else:
            metrics_buffer.additional_metrics[name].append(np.array(val))
    return metrics_buffer

  def _write_train_metrics(self):
    """Writes previous buffered train metrics."""
    if self._prev_buffered_train_metrics is None:
      # skip the first step so we can overlap I/O with next step.
      self._prev_buffered_train_metrics = self._buffered_train_metrics
      self._buffered_train_metrics = None
      return
    # increment the step by one for logging purpose, because train_step is not
    # incremented until the next model update.
    self._prev_buffered_train_metrics.step += 1
    self._write_metrics(self._prev_buffered_train_metrics)
    self._may_update_pbar(
        self._tqdm_train_metrics,
        step=self._prev_buffered_train_metrics.step,
        loss=self._prev_buffered_train_metrics.loss,
        step_time=self._prev_buffered_train_metrics.step_time_delta,
    )
    self._prev_buffered_train_metrics = self._buffered_train_metrics
    self._buffered_train_metrics = None

  def _write_metrics(self, metrics_buffer: MetricsBuffer):
    self._log_metrics(
        loss=metrics_buffer.loss,
        step=metrics_buffer.step,
        step_time_delta=metrics_buffer.step_time_delta,
        additional_metrics={
            k: np.mean(v)
            for k, v in metrics_buffer.additional_metrics.items()
        },
    )

  @contextlib.contextmanager
  def _switch_mode(self, mode: metrics_logger.Mode):
    original_mode = self._mode
    self._mode = mode
    try:
      yield
    finally:
      self._mode = original_mode

  @property
  def _tqdm_train_metrics(self) -> list[str]:
    return ["loss", "perplexity", "steps_per_sec", "learning_rate"]

  def _may_update_pbar(
      self,
      metrics: list[str],
      step: int | None = None,
      loss: ArrayLike | None = None,
      step_time: float | None = None,
  ):
    """Updates the progress bar with the given metrics if available."""
    if self._pbar is not None:
      self._pbar.update_metrics(metrics, self._mode, ndigits=3)
      self._pbar.update()

  def train(
      self,
      train_ds: Iterable[Any],
      num_train_sources: int ,
      eval_ds: Iterable[Any] | None = None,
      dev_ds: Iterable[Any] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Training loop."""
    micro_bs = self.fullConfig["batch_size"]
    subsel = self.fullConfig["subset_select"]["enabled"]
    mode = self.fullConfig["subset_select"]["mode"]
    
    self.meta = {
      "prev_utils": jnp.zeros((num_train_sources,), dtype=jnp.float32),
      "gain": jnp.zeros((num_train_sources,), dtype=jnp.float32)
    }
    cached_subsel = partial(subsel_utils.subset_select, self.model, num_train_sources, self.project, 
              self.optimizer_head, self.cache_train, self.cache_val, self.fullConfig)
    
    def get_train():
      if not subsel: return 1
      _ratio = self.fullConfig["subset_select"]["ratio"]
      if mode == "full": _ratio = 1
      _buffer = self.fullConfig["subset_select"]["buffer"]
      batch_to_buffer =int(_buffer*_ratio) 
      return batch_to_buffer
    
    batch_to_buffer = get_train()
    tosel = micro_bs*batch_to_buffer
    train_step, eval_step = self.jit_train_and_eval_step(skip_jit)
    train_step = partial(train_step, self.model, self.optimizer, self.optimizer_head, self.project)
    if not skip_jit:
      logging.info(
          "Training with mesh: %s",
          pxla.thread_resources.env.physical_mesh,
      )

    # if self.fullConfig["task_config"]["retrieval"]["eval"]["on_start"]:
    #   logging.info("Running evaluation before training starts.")
    #   if (self.eval_loss_fn is not None and eval_ds):
    #     seed = _step*100 + 42
    #     self._run_eval(eval_ds, eval_step, seed=seed, k=self.fullConfig["task_config"]["config"]["evalbs"])
      
    #   if (self.eval_fn):
    #     self.eval_fn(self.model, self._train_steps, self.metrics_logger)

    if self.config.max_steps is not None and self._pbar is None:
      self._pbar = progress_bar.ProgressBar(
          metrics_logger=self.metrics_logger,
          initial_steps=self._train_steps,
          max_steps=self.config.max_steps,
          description=self.config.pbar_description,
      )

    train_iterator = iter(train_ds)
    index = 0
    last_step_completion_time = time.perf_counter()
    step = -1
    with utils.time_measure("Train loop"):
      while True:
        step += 1
        self._prof.maybe_activate(self._iter_steps)
        with utils.time_measure("Train step"):
        # with jax.profiler.StepTraceAnnotation(
        #     "train", step_num=self._iter_steps
        # ):
          train_example = None
          try:
            train_example = next(train_iterator)
            if not self.is_managed_externally:
              # TODO(mridulsahu): Add support to restore the iterator state
              # instead of skipping the already trained examples.
              if index < self._iter_steps:
                # Skip the examples that are already trained.
                index += 1
                continue
            index += 1
          except StopIteration:
            pass

          if train_example is None:
            break

          # Stop training if max_steps is reached.
          if (
              self.config.max_steps is not None
              and self._train_steps >= self.config.max_steps
          ):
            break
          self.rng, key = jax.random.split(self.rng)
          
          train_example = self._shard_input(train_example)
          train_example = self.gen_model_input_fn(train_example)
          task_cfg = self.fullConfig["task_config"]["config"]
          grad_layer = task_cfg["grads"]["grad_layer"]

          if subsel and self.moments_notset and mode not in ["full", "random"]:
            self.moments_notset = False
            dummy = jax.tree.map(lambda x: jnp.zeros_like(x), train_example)
            dummy_grads, _ = subsel_utils.per_ex_grads(self.model, step, train_example, 
                                        task_cfg, grad_layer, key, moments=None)
            dummy_grads = jax.tree.map(lambda x: jnp.zeros_like(x.mean(0)), dummy_grads)
            self.moments = {
              "train": (dummy_grads, dummy_grads),
              "val": (dummy_grads, dummy_grads)
            }

    
          val_batch = None
          if subsel and dev_ds is not None:
            anchorbs = task_cfg["subsel"]["anchorbs"]
            val_batch = self.val_sample(dev_ds, 100*step + 42, anchorbs, n_val=-1)
          lr = self.schedule(self._train_steps)
          
          subsel_aux = {}
          if subsel:
            with utils.time_measure("Subsel", suppress_logging=False):
              mega_batch, moments, loss_mask, subsel_aux, meta = cached_subsel( self.moments, step,
                                                                                tosel, mode,
                                                                                train_example, val_batch,
                                                                                lr, key, self.meta)
            self.moments = moments
            self.meta = meta
          else:
            mega_batch = train_example
            loss_mask = jnp.ones((jax.tree.leaves(train_example)[0].shape[0],), dtype=jnp.float32)
          # jax.debug.visualize_array_sharding(train_example["negative"]["input_tokens"])
          # jax.debug.visualize_array_sharding(mega_batch["negative"]["input_tokens"])
          # batch_to_buffer = 1
          for i in range(batch_to_buffer):
          # for i in range(1):
            # i = 0
            train_example = jax.tree.map(lambda x: x[i*micro_bs: (i+1)*micro_bs], mega_batch)
            # jax.debug.print("{}", jax.tree.map(lambda x: x.shape, mega_batch))
            # train_example = mega_batch
            
            self._throttler.wait_for_next()

            with utils.time_measure("ModelStep", suppress_logging=True):
              meta = {
                "step": self._train_steps,
                "loss_mask": loss_mask,
                "rng": key
              }
              train_loss, aux = train_step(train_example, meta)

            print(aux, flush=True)

            current_time = time.perf_counter()
            step_time_delta = current_time - last_step_completion_time
            last_step_completion_time = current_time

            self._throttler.add_computation(train_loss)
            self._buffered_train_metrics = self._buffer_metrics(
                self._buffered_train_metrics,
                loss=train_loss,
                step=self._train_steps,
                step_time_delta=step_time_delta,
                additional_metrics={**aux, **subsel_aux} if i == 0 else aux
            )
            self._iter_steps += 1

            if (
                self._iter_steps
                % self.config.get_with_default("gradient_accumulation_steps", 1)
                == 0
            ):
              self._train_steps += 1
              self._write_train_metrics()

              # Checkpoint frequency is configured by checkpointing_options.
              self.checkpoint_manager.save(
                  self._train_steps,
                  self.model,
                  save_only_lora_params=self._lora_enabled,
              )
              to_eval = self._train_steps % self.config.eval_every_n_steps == 0

              if (self.eval_loss_fn is not None and eval_ds and to_eval):
                seed = self._train_steps * 100 + 42
                self._run_eval(eval_ds, eval_step, seed=seed, k=self.fullConfig["task_config"]["config"]["evalbs"])

              if (self.eval_fn and to_eval):
                self.eval_fn(self.model, self._train_steps, self.metrics_logger)

        self._prof.maybe_deactivate(self._iter_steps)

    self._throttler.wait_for_all()
    if not self.is_managed_externally:
      self.close()

  def _save_last_checkpoint(self):
    last_saved_step = self.checkpoint_manager.latest_step()
    if last_saved_step is None or last_saved_step < self._train_steps:
      self.checkpoint_manager.save(
          self._train_steps,
          self.model,
          save_only_lora_params=self._lora_enabled,
          force=True,
      )

  @property
  def train_steps(self) -> int:
    """Returns the number of train steps taken."""
    return self._train_steps

  @property
  def iter_steps(self) -> int:
    """Returns the number of iterator steps taken."""
    return self._iter_steps

  def close(self):
    """Closes the trainer and its associated resources.

    This includes writing any buffered metrics, saving the last checkpoint,
    and closing the checkpoint manager and metrics logger.
    """
    self._write_train_metrics()
    self._save_last_checkpoint()
    self.checkpoint_manager.close()
    self.metrics_logger.close()
    if self._pbar is not None:
      self._pbar.close()
      self._pbar = None

  def val_sample(self, iterable, seed: int, batch_size: int, n_val: int = -1):
      import random
      rng = random.Random(seed)

      if self.eval_ds_back is None:
          cache = []
          for batch in iterable:
              first_key = next(k for k in batch.keys() if k != "meta")
              B = batch[first_key].shape[0]

              for i in range(B):
                  ex = {}
                  for k, v in batch.items():
                      if k == "meta":
                          ex["meta"] = {mk: mv[i] for mk, mv in v.items()}
                      else:
                          ex[k] = v[i]
                  cache.append(ex)

          rng.shuffle(cache)
          if n_val > 0: cache = cache[:n_val]
          self.eval_ds_back = cache

      # Sample a batch from the cached examples
      n = len(self.eval_ds_back)
      if n == 0:
          raise ValueError("eval_ds_back is empty; iterable produced no examples.")

      idxs = rng.sample(range(n), k=min(batch_size, n))
      exs = [self.eval_ds_back[i] for i in idxs]
      example0 = exs[0]

      eval_ex = {}
      for k, v0 in example0.items():
          if k == "meta":
              eval_ex["meta"] = {
                  mk: jnp.stack([ex["meta"][mk] for ex in exs], axis=0)
                  for mk in v0.keys()
              }
          else:
              eval_ex[k] = jnp.stack([ex[k] for ex in exs], axis=0)

      eval_ex = self._shard_input(eval_ex)
      eval_ex = self.gen_model_input_fn(eval_ex)
      return eval_ex

      
  def _reservoir_sample(self, iterable: Iterable[Any], k: int, rng) -> list[Any]:
    """Returns k random elements from an iterable in one pass (reservoir sampling)."""
    sample: list[Any] = []
    for i, x in enumerate(iterable):
      if i < k:
        sample.append(x)
      else:
        j = rng.randint(0, i)
        if j < k:
          sample[j] = x
    return sample

  def _run_eval_dict(
      self, eval_ds_dict, eval_step_fn, 
  ):
    """
    Evaluate multiple datasets.
    - Per-dataset loss: mean over that dataset's batches.
    - Overall loss: mean of per-dataset means (equal dataset weighting).

    """
    step = self._train_steps
    from tqdm import tqdm
    logging.info("Running evaluation on train step %d.", step)

    per_ds_mean_loss: dict[str, float] = {}

    for ds_name, eval_ds in tqdm(eval_ds_dict.items()):
      ds_loss_sum = 0.0
      ds_num_batches = 0

      for batch in tqdm(eval_ds, desc=ds_name, leave=False):
        batch = self._shard_input(batch)
        loss, _ = eval_step_fn(self.model, batch)
        loss = jax.lax.stop_gradient(loss)

        ds_loss_sum += float(loss)
        ds_num_batches += 1

      if ds_num_batches == 0:
        logging.warning("No eval examples found for dataset '%s'. Skipping.", ds_name)
        continue

      mean_loss = ds_loss_sum / ds_num_batches
      per_ds_mean_loss[ds_name] = mean_loss

      self.metrics_logger.log(f"loss", mean_loss, ds_name, step)

    if not per_ds_mean_loss:
      logging.warning("No eval examples found across all datasets.")
      return {}, float("nan")

    overall_mean_loss = sum(per_ds_mean_loss.values()) / len(per_ds_mean_loss)

    self.metrics_logger.log(f"loss", overall_mean_loss, "overall", step)
    logging.info(
        "Train step %d eval loss (equal-weight mean over datasets): %f",
        step,
        overall_mean_loss,
    )


  def _run_eval(
      self,
      eval_ds: Iterable[Any],
      eval_step_fn: Callable[..., Any],
      seed=None,
      k: int = 8,
  ) -> None:
    """Runs evaluation loop."""
    logging.info("Running evaluation on train step %d.", self._train_steps)
    if isinstance(eval_ds, dict):
      logging.info("Got dict as ds %d.", self._train_steps)
      self._run_eval_dict(eval_ds, eval_step_fn)
      return
    # --- NEW: sample k items from the iterable (fresh every call) ---
    if k is not None and k > 0:
      # if you want deterministic per-step, e.g. seed=self._train_steps
      # Note: this consumes eval_ds once to choose the subset, but we only *evaluate* k items
      import random
      rng = random.Random(seed)
      eval_batches = self._reservoir_sample(eval_ds, k, rng)
      eval_iterator = iter(eval_batches)
    else:
      eval_iterator = iter(eval_ds)
    step = -1
    with self._switch_mode(metrics_logger.Mode.EVAL):
      eval_loss, eval_steps = 0, 0
      while True:
        step += 1
        try:
          eval_example = next(eval_iterator)
        except StopIteration:
          eval_example = None
        if eval_example is None:
          break
        eval_example = self._shard_input(eval_example)
        loss, aux = eval_step_fn(self.model, eval_example)
        loss = jax.lax.stop_gradient(loss)
        self._buffered_eval_metrics = self._buffer_metrics(
            self._buffered_eval_metrics,
            loss=loss,
            step=self._train_steps,
        )
        eval_loss += loss
        eval_steps += 1

      if eval_steps == 0:
        logging.warning(
            "No eval examples found. Skipping eval metrics logging."
        )
        return

      self._write_metrics(self._buffered_eval_metrics)
      logging.info(
          "Train step %d eval loss: %f - eval perplexity: %f",
          self._train_steps,
          self.metrics_logger.get_metric("loss", "eval"),
          self.metrics_logger.get_metric("perplexity", "eval"),
      )
      self._buffered_eval_metrics = None


'''
adam
bs128
llama 3b
eval
100 iters
1e-2 20 iters
sim mat

'''