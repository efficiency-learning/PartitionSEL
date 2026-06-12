# PartitionSel

**Joint data subset selection for efficient LLM instruction tuning.**

PartitionSel fine-tunes a decoder LLM (LoRA or full) while training on a *selected
subset* of each incoming data batch instead of the whole batch. At every step it
computes per-example gradient features, scores candidate examples against a held-out
anchor set, and keeps the most useful fraction — so the model sees more data but
trains on a compute-bounded, high-utility subset.

The selection objective is configurable (see [Subset selection modes](#subset-selection-modes)),
with `greats` and the weighted `joint` / `iwd` partition-selection objectives as the
primary methods.

---

## Requirements

- Linux + NVIDIA GPU (CUDA 12). Single- or multi-GPU.
- Python 3.11 or 3.12.
- A Hugging Face account/token (models + datasets are pulled from the Hub).

---

## Setup

```bash
conda create -n partitionsel python=3.11 -y
conda activate partitionsel

# Core dependencies (GPU / CUDA 12)
pip install -U "jax[cuda12]" flax optax chex qwix orbax-checkpoint grain \
    transformers datasets sentencepiece huggingface_hub kagglehub \
    omegaconf tensorboardX tqdm python-dotenv

# PyTorch is used only by the data loaders — the CPU build is enough
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install this package in editable mode (code only; keep the JAX build above)
pip install -e . --no-deps
```

> The training/eval CLI is exposed as the `tunix.cli` module namespace, so after the
> editable install you launch it with `python -m tunix.cli.peft_main` / `python -m tunix.cli.generate`.

### Credentials

The example scripts read credentials from the environment. Export them in your shell
(do **not** hardcode them in tracked files):

```bash
export HF_TOKEN="<your-hf-token>"
export KAGGLE_USERNAME="<your-kaggle-username>"   # only needed for Kaggle-hosted models
export KAGGLE_KEY="<your-kaggle-key>"
```

---

## Data

Datasets are downloaded automatically from the Hub on first run. Set the
`dataset_name` argument to select one:

| `dataset_name`              | Task                          |
| :-------------------------- | :---------------------------- |
| `meta-math/MetaMathQA`      | Math instruction tuning       |
| `zjunlp/Mol-Instructions`   | Molecular/biomolecule instruction tuning |

---

## Training

The ready-to-run example launchers live in `examples/sft/mtnt/`. The main one is
[`run_llama3.2_3b_ift.sh`](examples/sft/mtnt/run_llama3.2_3b_ift.sh) (Llama-3.2-3B +
LoRA on Mol-Instructions with `greats` selection):

```bash
bash examples/sft/mtnt/run_llama3.2_3b_ift.sh
```

For a fast end-to-end check (Qwen2.5-0.5B, 1 step, selection off), use the smoke test:

```bash
bash examples/sft/mtnt/run_mol_smoke_train.sh
```

### Launching directly

Under the hood the scripts call:

```bash
python3 -m tunix.cli.peft_main \
  base_config.yaml \
  model_config.model_name=llama3.2-3b \
  model_config.model_id="meta-llama/Llama-3.2-3B" \
  model_config.model_source="huggingface" \
  model_config.lora_config='{"module_path":".*q_proj|.*k_proj|.*v_proj|.*gate_proj|.*down_proj|.*up_proj","rank":16,"alpha":96.0}' \
  dataset_name="zjunlp/Mol-Instructions" \
  task_config.task=ift \
  task_config.config="examples/sft/mtnt/task_config.yaml" \
  batch_size=32 \
  max_target_length=512 \
  subset_select.enabled=true \
  subset_select.mode=greats \
  subset_select.ratio=0.125 \
  subset_select.buffer=8 \
  optimizer_config.opt_type="adamw" \
  optimizer_config.learning_rate=2e-4 \
  training_config.max_steps=1024 \
  training_config.checkpoint_root_directory="examples/sft/mtnt/ckpts/my-run"
```

Defaults come from [`tunix/cli/base_config.yaml`](tunix/cli/base_config.yaml); any field
can be overridden on the command line with `dotted.key=value`.

### Key subset-selection knobs

| Argument                  | Meaning                                                                 |
| :------------------------ | :---------------------------------------------------------------------- |
| `subset_select.enabled`   | Turn selection on/off (off = standard full-batch fine-tuning).          |
| `subset_select.mode`      | Selection objective — see table below.                                  |
| `subset_select.ratio`     | Fraction of the candidate pool kept per step (e.g. `0.125`).            |
| `subset_select.buffer`    | Candidate-pool multiplier: each step loads `batch_size × buffer` examples and selects from them. |

### Subset selection modes

Set via `subset_select.mode=<mode>`:

| Mode       | Description                                                              |
| :--------- | :---------------------------------------------------------------------- |
| `greats`   | GREATS-style greedy selection (score + pairwise interaction).           |
| `joint`    | Weighted partition-selection objective solved with APGD (curriculum-aware budgets). |
| `iwd`      | Independent weighted domain-wise selection (per-domain block-diagonal). |
| `gradnorm` | Top-k by per-example gradient norm.                                     |
| `facloc`   | Facility-location coverage over the anchor similarities.                |
| `uniprot`  | Fair-OT greedy selection.                                               |
| `random`   | Random subset (baseline).                                               |
| `full`     | Keep the whole batch (baseline; equivalent to `enabled=false`).         |

---

## `task_config.yaml`

The `task_config.config` argument points at a YAML (default:
[`examples/sft/mtnt/task_config.yaml`](examples/sft/mtnt/task_config.yaml)) that
controls *how* selection is computed, independent of the model/optimizer args. Main
blocks:

- **`grads`** — per-example gradient features used for scoring:
  - `grad_layer`: which transformer layers to take gradients from (e.g. `[27,26,25]`).
  - `use_lora`: gradients w.r.t. LoRA params (`true`) or full weights (`false`).
  - `dimred` / `dimred_dim`: optional FFT-based dimensionality reduction of the gradient
    features (random projection to `dimred_dim`) to cut selection cost/memory.
  - `chunk_size`, `normalize`: micro-batching and normalization for the gradient pass.
- **`subsel`** — selection hyperparameters: validation/anchor settings (`val_anchors`,
  `anchorbs`), domain-wise options (`domainwise`, `apdg_iters`), fair-OT params
  (`reg`, `iters`), and minority-class handling (`minority_full`, `minority_classes`).
- **`curricullum`** — curriculum budget schedule used by `joint` (`beta`, `lamb`).
- **`domain_weights`** — optional domain reweighting of the training mixture.

---

## Inference / evaluation

Evaluation generates from saved checkpoints with `tunix.cli.generate`. The launcher
[`run_mol_smoke_eval.sh`](examples/sft/mtnt/run_mol_smoke_eval.sh) is the template:

```bash
bash examples/sft/mtnt/run_mol_smoke_eval.sh
```

It is driven by environment variables:

| Env var           | Meaning                                                            |
| :---------------- | :----------------------------------------------------------------- |
| `EVAL_BENCHMARKS` | Comma-separated benchmarks: `mol`, `colm`, `legalbench`.           |
| `EVAL_STEPS`      | Checkpoint step(s) to evaluate: `4096`, `1024,2048`, or `all`.     |
| `EVAL_SPLIT`      | Which split to score (default `all`).                              |

Point `training_config.checkpoint_root_directory` at the directory your training run
wrote to so the matching checkpoints are restored.

---

## Outputs

Each run (under `examples/sft/mtnt/`) writes:

- `ckpts/<EXPNAME>/` — LoRA/model checkpoints.
- `tensorboard/<EXPNAME>/` — TensorBoard logs (`tensorboard --logdir examples/sft/mtnt/tensorboard`).
- `results/<EXPNAME>/logs.log` — stdout/stderr of the run.

---

## Repository layout

```
tunix/cli/            # CLI entry points: peft_main (train), generate (eval), base_config.yaml
tunix/sft/            # training loop (subset_trainer) + selection (subsel/, subsel_utils.py)
tunix/sft/subsel/     # subset-selection core: subsel.py, grads.py, dimred.py, utils.py
tunix/sft/eval/       # evaluation harnesses (mol, colm, legalbench, mmlu, tydiqa)
tunix/examples/data/  # dataset loaders (ift_dataset.py + ift/ for MetaMathQA/Mol/...)
tunix/models/         # model definitions (llama3, qwen2, qwen3, gemma, gemma3)
examples/sft/mtnt/    # run scripts + task_config.yaml
```
