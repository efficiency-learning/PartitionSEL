from absl import app
import jax
from tunix.generate import sampler
from tunix.cli import config
from tunix.sft import checkpoint_manager
from tunix.cli.utils import model as model_lib
import os, json
import datasets
from functools import partial
from tqdm import tqdm

import tunix.sft.eval.chat_templates as chat_templates
import tunix.sft.eval.colm.run_eval as colm_eval
import tunix.sft.eval.mol.molecule.evaluate as mol_eval
from tunix.sft.eval.legalbench.run_eval import run_eval as legalbench_eval
import os
from tabulate import tabulate
import csv
from collections import defaultdict

class PeftPipeline(config.HyperParameters):

  def get_model(self, load_fresh=False):
    mesh: jax.sharding.Mesh = self.create_mesh('model_config')
    print("config", self.config["training_config"])
    ckpt_root = self.config["training_config"]["checkpoint_root_directory"]
    
    model, tokenizer_path = model_lib.create_model(
      self.config['model_config'], self.config['tokenizer_config'], mesh
    )
    if model is None:
      raise ValueError('model is None')
    tokenizer = model_lib.create_tokenizer(
      self.config['tokenizer_config'], tokenizer_path
    )
    if not load_fresh:
        print("LOADING FROM CHECKPOINT")
        ckpt_manager = checkpoint_manager.CheckpointManager(
            root_directory=ckpt_root
        )
        ckpt_manager.maybe_restore(model, self.config["inference_restore_step"], restore_only_lora_params=True)
    return model, mesh, tokenizer

  def run_inference(self, my_sampler, max_generation_steps, 
                    max_prompt_length, candidate_token_ids, batch_size, inputs):
    mysamp = my_sampler
    
    N = len(inputs)
    if N == 0: raise ValueError("inputs is empty")

    num_batches = (N + batch_size - 1) // batch_size
    pad = num_batches * batch_size - N

    inputs_padded = inputs
    logits_chunks = []
    out_strs = []
    PROC = max_generation_steps == 1

    if pad > 0:
      reps = (pad + N - 1) // N
      inputs_padded = inputs + (inputs * reps)[:pad]

    for i in range(0, len(inputs_padded), batch_size):
      batch = inputs_padded[i : i + batch_size]
      outs = mysamp(
          batch,
          max_generation_steps=max_generation_steps,
          echo=False,
          return_logits=True,
          max_prompt_length=max_prompt_length,
      )

      out_strs.extend(outs.text)
      if PROC: logits_chunks.extend(outs.logits)

    if PROC:
        batch_logits = jax.numpy.concat(logits_chunks, axis=0)[:N]
        batch_probs = jax.scipy.special.softmax(batch_logits, axis=-1)
        print("lens", len(inputs), len(inputs_padded), batch_probs.shape)

        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]

        batch_prediction_indices = jax.numpy.argmax(batch_probs, axis=-1)

        return out_strs, batch_prediction_indices.tolist()
    return out_strs, None


def _list_ckpt_steps(ckpt_root):
  """Return sorted list of checkpoint step numbers found in ckpt_root."""
  steps = []
  for entry in os.scandir(ckpt_root):
    if entry.is_dir() and entry.name.isdigit():
      steps.append(int(entry.name))
  return sorted(steps)


def main(argv, **kwargs):
  pipeline = PeftPipeline(argv, **kwargs)
  ckpt_root = pipeline.config["training_config"]["checkpoint_root_directory"]

  eval_steps_env = os.environ.get("EVAL_STEPS", "").strip()
  assert eval_steps_env, "EVAL_STEPS env var is required (e.g. '4096', '1024,2048,4096', or 'all')"
  if eval_steps_env == "all":
    steps = _list_ckpt_steps(ckpt_root)
  else:
    steps = sorted(int(s) for s in eval_steps_env.split(","))
  print(f"Eval steps: {steps}")

  eval_split = os.environ.get("EVAL_SPLIT", "all").strip()
  print(f"Eval split: {eval_split}")

  model, mesh, tokenizer = pipeline.get_model(load_fresh=True)
  ckpt_mgr = checkpoint_manager.CheckpointManager(root_directory=ckpt_root)

  for step_idx, step in enumerate(steps):
    print(f"\n{'='*60}")
    print(f"EVALUATING checkpoint {step}  ({step_idx+1}/{len(steps)})")
    print(f"{'='*60}")
    if step != -1:
      ckpt_mgr.maybe_restore(model, step, restore_only_lora_params=True)
    pipeline.config["inference_restore_step"] = step

    benchmarks_env = os.environ.get("EVAL_BENCHMARKS", "colm").strip()
    benchmarks = [b.strip() for b in benchmarks_env.split(",")]
    print(f"Eval benchmarks: {benchmarks}")

    if "colm" in benchmarks:
      COLM_eval(pipeline, model, mesh, tokenizer, eval_split=eval_split)
    if "legalbench" in benchmarks:
      legalbench_eval(pipeline, model, mesh, tokenizer, eval_split=eval_split, summarize_fn=summarize_results)
    if "mol" in benchmarks:
      MOL_eval(pipeline, model, mesh, tokenizer)


def COLM_eval(pipeline, model, mesh, tokenizer, eval_split="all"):
  exp_name = pipeline.config["training_config"]["checkpoint_root_directory"]
  ckpt_num = pipeline.config["inference_restore_step"]
  exp_name = os.path.basename(exp_name.rstrip("/"))
  tokenizer._tokenizer.chat_template = chat_templates.qwen2_5_template

  base_args = {
    "max_prompt_tokens": 1024,
    "max_generation_steps": 256,
  }
  batch_size = 16

  split_tag = f"_{eval_split}" if eval_split != "all" else ""
  out_dir = f"/root/prayas/ret-subset/examples/sft/mtnt/results/{exp_name}/{ckpt_num}{split_tag}"
  data_root = "/root/prayas/CoLM/math_eval/dataset"
  os.makedirs(out_dir, exist_ok=True)
  
  with mesh:
    mysamp = sampler.Sampler(
        model,
        tokenizer,
        sampler.CacheConfig(
            cache_size=2048,
            num_layers=model.config.num_layers,
            num_kv_heads=model.config.num_kv_heads,
            head_dim=model.config.head_dim,
        ),
    )
    
    inference_function = partial(pipeline.run_inference, mysamp, base_args["max_generation_steps"], 
                                 base_args["max_prompt_tokens"], [], batch_size)
    inf2 = lambda batch: inference_function(batch)[0]
    maths = [
        'mmlu_elementary-mathematics', 
        'mmlu_high-school-mathematics', 
        'mmlu_college-mathematics', 
        'mmlu_abstract-algebra', 
        'mmlu_formal-logic']
    numglue = [
        'numglue_Type_2', 'numglue_Type_4', 'numglue_Type_3', 'numglue_Type_8', 'numglue_Type_1'
    ]
    colm_dataset = ["numglue", "mmlu_mathematics", "gsm8k",  "svamp", "simuleq", "deepmind", "aqua", "sat"]
    # colm_dataset = ["mmlu_mathematics",  "aqua", "sat"]
    flan_tag = ""
    all_res = []
    for eval_ds in colm_dataset:
      num_shots = 0
      if "mmlu" in eval_ds: num_shots = 2
      if "aqua" in eval_ds: num_shots = 2
      if "sat" in eval_ds: num_shots = 2
      print(f"TESTING {eval_ds} at {num_shots} shots")
      fname = f"{eval_ds}_acc.jsonl"
      if "pot" in flan_tag: fname = f"{eval_ds}_pot_acc.jsonl"
      out_path = os.path.join(out_dir, fname)
      res = colm_eval.run_eval(inf2, tokenizer._tokenizer, 
                               base_args["max_prompt_tokens"], out_path, data_root, eval_ds, num_shots, 
              batch_size, stem_flan_type=flan_tag, cot_backup=True, debug=True, split=eval_split, dev_frac=0.2)
      all_res.append(res)

  summarize_results(all_res, f"{out_dir}/results.csv")


def summarize_results(results, csv_path):
    

    avg_acc = sum(r["accuracy"] for r in results) / len(results)

    grouped = defaultdict(list)
    for r in results:
        grouped[r["shots"]].append(r["accuracy"])
    group_avgs = {shots: sum(vals) / len(vals) for shots, vals in grouped.items()}

    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        fieldnames = ["dataset", "accuracy", "shots"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(results)

        for shots, avg in sorted(group_avgs.items(), key=lambda x: x[0]):
            writer.writerow({"dataset": f"AVERAGE_SHOT_{shots}", "accuracy": avg, "shots": shots})

        writer.writerow({"dataset": "AVERAGE", "accuracy": avg_acc, "shots": ""})

    display_rows = results.copy()
    for shots, avg in sorted(group_avgs.items(), key=lambda x: x[0]):
        display_rows.append({"dataset": f"AVERAGE_SHOT_{shots}", "accuracy": avg, "shots": shots})
    display_rows.append({"dataset": "AVERAGE", "accuracy": avg_acc, "shots": ""})

    print("\nAverage Accuracy: {:.4f}\n".format(avg_acc))
    print(tabulate(display_rows, headers="keys", tablefmt="github", floatfmt=".4f"))
    print(csv_path)

    return avg_acc, group_avgs


def MOL_eval(pipeline, model, mesh, tokenizer):
  exp_name = pipeline.config["training_config"]["checkpoint_root_directory"]
  ckpt_num = pipeline.config["inference_restore_step"]
  exp_name = os.path.basename(exp_name.rstrip("/"))
  tokenizer._tokenizer.chat_template = chat_templates.qwen2_5_template
  tok = tokenizer._tokenizer
  base_args = {
    "max_prompt_tokens": int(os.environ.get("MOL_MAX_PROMPT_TOKENS", "896")),
    "max_generation_steps": int(os.environ.get("MOL_MAX_GENERATION_STEPS", "128")),
  }
  num_test = int(os.environ.get("MOL_NUM_TEST", "1000"))
  batch_size = int(os.environ.get("MOL_BATCH_SIZE", "16"))
  ckpt_root = pipeline.config["training_config"]["checkpoint_root_directory"].rstrip("/")
  root = os.path.join(os.path.dirname(os.path.dirname(ckpt_root)), "results")
  out_dir = f"{root}/{exp_name}/{ckpt_num}/MOL_{num_test}"
  os.makedirs(out_dir, exist_ok=True)
  
  with mesh:
    mysamp = sampler.Sampler(
        model,
        tokenizer,
        sampler.CacheConfig(
            cache_size=1024,
            num_layers=model.config.num_layers,
            num_kv_heads=model.config.num_kv_heads,
            head_dim=model.config.head_dim,
        ),
    )
    
    inference_function = partial(pipeline.run_inference, mysamp, base_args["max_generation_steps"], 
                                 base_args["max_prompt_tokens"], [], batch_size)
    inf2 = lambda batch: inference_function(batch)[0]
    MOL_GEN(inf2, num_test, tok, out_dir, batch_size=batch_size)
    mol_eval.run_eval(out_dir)


def MOL_GEN(inference_fn, num_test, tokenizer, out_dir, batch_size=32):
    os.makedirs(out_dir, exist_ok=True)

    cfgs = ['Molecule-oriented Instructions', 'Protein-oriented Instructions', 'Biomolecular Text Instructions']
    ds = datasets.load_dataset("zjunlp/Mol-Instructions", cfgs[0], trust_remote_code=True)

    keys = [
        "description_guided_molecule_design",
        "forward_reaction_prediction",
        "reagent_prediction",
        "retrosynthesis"
    ]
    dsdict = {}
    dsdict_demos = {}
    n_shots = 2

    for i,key in enumerate(keys):
        t = ds[key].train_test_split(test_size=num_test, seed=42, shuffle=True)
        dsdict[key] = t["test"]
        dsdict_demos[key] = {"instruction": [], "input": [], "output": []}
        if n_shots > 0:
          dsdict_demos[key] = t["train"].train_test_split(test_size=n_shots, seed=42, shuffle=True)["test"]

    for key, _ in dsdict.items():
      dataset = dsdict[key]
      demos = dsdict_demos[key]
      print("PROCESSING", key, len(dataset), len(dataset) // batch_size)
      out_path = f"{out_dir}/{key}.jsonl"

      with open(out_path, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(dataset), batch_size), desc=key):
          batch = dataset[start:start+batch_size]
          prompts = chat_templates.promptify_alpaca([], batch)

          ground = batch["output"]
          preds = inference_fn(prompts)

          for d, g, p in zip(prompts, ground, preds):
            record = {
              "description": d,
              "ground_truth": g,
              "output": p,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == '__main__':
  app.run(main)
