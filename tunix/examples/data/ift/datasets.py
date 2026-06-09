from collections import Counter

import datasets
import numpy as np
from datasets import Dataset
from tqdm import tqdm

import tunix.sft.eval.chat_templates as chat_templates
from tunix.sft.eval.colm import prompt_utils as colm_eval
from tunix.sft.eval.data_selection import get_validation_dataset

from .configs import (
    LAWINSTRUCT_DICT,
    LEGALBENCH_SOURCE_MODE,
    LEGALBENCH_TASK_TYPE_MAP,
    LEGALBENCH_TASKS,
)


def process_data(dataset, data_dict, verbose=True, num_proc=50):
  """raw_sources: list[str] -> (data_sources: list[int], all_data_sources: list[str], num_sources: int)"""
  key_instruction = data_dict["instruction"]
  key_input = data_dict.get("input", None)
  key_response = data_dict["response"]
  key_source = data_dict["source"]
  ds_name = data_dict["ds_name"]

  dataset = dataset.filter(lambda ex: bool(ex[key_response]), num_proc=num_proc,
                           desc="Filtering empty responses")

  # Vectorized column access instead of row-by-row iteration
  raw_sources = dataset[key_source]

  all_data_sources = sorted(set(raw_sources))
  src2id = {s: i for i, s in enumerate(all_data_sources)}
  if verbose:
    print("#"*50)
    print(src2id)
    print("#"*50)
  data_sources = [src2id[s] for s in raw_sources]
  num_sources = len(all_data_sources)

  def get_msg(ex):
    if key_input is None:
      msg = [
        {"role": "user", "content": ex[key_instruction]},
        {"role": "assistant", "content": ex[key_response]},
      ]
    else:
      msg = [
        {"role": "user", "content": ex[key_instruction]},
        {"role": "user", "content": ex[key_input]},
        {"role": "assistant", "content": ex[key_response]},
      ]

    return msg

  def proc(ex, idx):
    ret = {
      "dataset": ds_name,
      "id": f"{idx}",
      "messages": get_msg(ex),
      "source": data_sources[idx]
    }
    return ret
  dataset = dataset.map(
      proc,
      with_indices=True,
      num_proc=num_proc,
      desc="Formatting to chat messages",
  )
  return dataset, src2id, num_sources


def get_eval_greats(tokenizer, config, max_target_length):
  ddir = "/home/aiscuser/prayas/temp/data"
  n_val = 2000
  task = "mmlu"
  subject =  [
    "abstract_algebra",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics"
  ]
  print("Subjects", subject)
  wordy_dev = config["task_config"]["config"]["wordydev"]
  eval_ds = get_validation_dataset.get_dataset(
      task,
      data_dir=ddir,
      tokenizer=tokenizer,
      max_length=max_target_length,
      validation=True,
      k=n_val,
      subject=subject,
      append_choice_text="eval" in wordy_dev
  )
  eval_ds.set_format(type="pt")
  
  dev_ds = get_validation_dataset.get_dataset(
      task,
      data_dir=ddir,
      tokenizer=tokenizer,
      max_length=max_target_length,
      validation=True,
      k=n_val,
      subject=subject,
      append_choice_text="dev" in wordy_dev
  )
  dev_ds.set_format(type="pt")

  return eval_ds, dev_ds


def convert_to_chatformat(user, asst, idx):
  ret = {
    "dataset": "tiger",
    "id": f"{idx}",
    "messages": [
      {"role": "user", "content": user},
      {"role": "assistant", "content": asst},
    ],
  }
  return ret


def get_colm(tokenizer, shots, max_seq_length, mask_input, cands=[]):
  if not len(cands):
    # FIXME: we skip POT demonstrations for now, since 
    # code assisted eval is out of scope atm.
    cands = [k for k in  colm_eval.get_ex_dict("").keys() if "_pot" not in k]
  src2id = {k: id for id, k in enumerate(set(sorted(cands)))}
  dataset = {"input_ids": [], "attention_mask": [], "labels": [], "source": []}
  i = 0
  for ds in cands:
    exs = colm_eval.get_examples(ds, shots, "")
    for ex in exs:
      ques, resp = ex[0], ex[1]
      ret, text = chat_templates.tokenize_prompt_alpaca(tokenizer, [], ques, 
        max_seq_length, resp=resp, mask_value=-100, return_text=True, mask_input=mask_input)
      ret = {k: v.flatten() for k,v in ret.items()}
      if i < 5:
        print(">>>val ques", ques)
        print(">>>val resp", resp)
        print(">>>val text", text)
        print(">>>Val ret", ret)
        print("############")
      i += 1

      dataset["source"].append(src2id[ds])
      for k, v in ret.items():
        dataset[k].append(v)
  dataset = Dataset.from_dict(dataset)

  dataset.set_format(type="pt")
  return dataset


def get_legalbench(tokenizer, max_seq_length, mask_input, tasks=None,
                   split="train", cap_per_task=None):
  """Load a curated subset of LegalBench (nguha/legalbench) as eval/dev data.

  Uses LegalBench's train split by default (held-out from train), keeping
  the test split fully held-out for final evaluation.
  """
  if tasks is None:
    tasks = LEGALBENCH_TASKS

  # Build source mapping
  if LEGALBENCH_SOURCE_MODE == "task_type":
    def _get_type(t):
      for prefix, typ in LEGALBENCH_TASK_TYPE_MAP.items():
        if t.startswith(prefix):
          return typ
      return "other"
    type_names = sorted(set(_get_type(t) for t in tasks))
    src2id = {t: type_names.index(_get_type(t)) for t in tasks}
  else:
    src2id = {t: i for i, t in enumerate(sorted(tasks))}

  dataset = {"input_ids": [], "attention_mask": [], "labels": [], "source": []}
  loaded_tasks = []
  task_counts = {}       # task_name -> number of examples kept
  skipped_tasks = []     # tasks that failed to load

  for task_name in tasks:
    try:
      ds = datasets.load_dataset("nguha/legalbench", task_name,
                                  split=split, trust_remote_code=True)
    except Exception as e:
      print(f"Warning: skipping LegalBench task {task_name}: {e}")
      skipped_tasks.append(task_name)
      continue

    if cap_per_task is not None:
      ds = ds.shuffle(seed=42).select(range(min(cap_per_task, len(ds))))

    loaded_tasks.append(task_name)
    count_before = len(dataset["input_ids"])
    for i, ex in enumerate(ds):
      ques = ex.get("text", "")
      resp = ex.get("answer", "")
      if not resp:
        continue

      ret, text = chat_templates.tokenize_prompt_alpaca(
          tokenizer, [], ques, max_seq_length,
          resp=resp, mask_value=-100, return_text=True, mask_input=mask_input)
      ret = {k: v.flatten() for k, v in ret.items()}

      if i < 2 and task_name == loaded_tasks[0]:
        print(f">>>legalbench [{task_name}] ques:", ques[:200])
        print(f">>>legalbench [{task_name}] resp:", resp)
        print(f">>>legalbench [{task_name}] text:", text[:300])
        print("############")

      dataset["source"].append(src2id[task_name])
      for k, v in ret.items():
        dataset[k].append(v)

    task_counts[task_name] = len(dataset["input_ids"]) - count_before

  # ── Stats summary ──
  total = len(dataset["input_ids"])
  print("\n" + "=" * 60)
  print("LEGALBENCH STATS")
  print("=" * 60)
  print(f"{'Task':<55} {'Count':>5}")
  print("-" * 60)
  for t in sorted(task_counts):
    cap_flag = ""
    if cap_per_task is not None and task_counts[t] >= cap_per_task:
      cap_flag = " [CAPPED]"
    print(f"  {t:<53} {task_counts[t]:>5}{cap_flag}")
  print("-" * 60)
  print(f"  {'TOTAL':<53} {total:>5}")
  print(f"  Tasks loaded: {len(loaded_tasks)}/{len(tasks)}")
  if skipped_tasks:
    print(f"  Skipped tasks: {skipped_tasks}")
  print(f"  Source mode: {LEGALBENCH_SOURCE_MODE}")
  print(f"  Split: {split} | mask_input: {mask_input}")
  if cap_per_task is not None:
    print(f"  Cap per task: {cap_per_task}")
  print("=" * 60 + "\n")

  ds_out = Dataset.from_dict(dataset)
  ds_out.set_format(type="pt")
  return ds_out


def get_lawinstruct(
    hf_config="all_english-1_english",
    cap_per_group=5000,
    cap_key="dataset_name",
    skip_task_types=None,
):
  """Load and prepare LawInstruct training data.

  Args:
    hf_config: HF dataset config name.
    cap_per_group: Per-group cap. None for no cap.
      If < 1, treated as a fraction of that group's total count.
      If >= 1, treated as an absolute count.
    cap_key: Column to group by when capping.
    skip_task_types: set/list of task_type values to exclude.

  Returns (train_ds, src2id, num_train_sources).
  """
  lawinstruct_config = hf_config
  lawinstruct_cap_per_ds = cap_per_group
  lawinstruct_cap_key = cap_key

  print(f"Loading LawInstruct config={lawinstruct_config}, "
        f"cap_per_ds={lawinstruct_cap_per_ds}, cap_key={lawinstruct_cap_key}")
  raw = datasets.load_dataset(
      "lawinstruct/lawinstruct", lawinstruct_config,
      split="train", trust_remote_code=True
  )
  raw = raw.shuffle(seed=42)

  # Filter out examples with empty answers
  raw = raw.filter(lambda ex: bool(ex["answer"] and ex["answer"].strip()), num_proc=10)

  # Filter to English-only (prompt AND answer must be English)
  total_pre_lang = len(raw)
  raw = raw.filter(
      lambda ex: ex.get("prompt_language", "") == "en" and ex.get("answer_language", "") == "en",
      num_proc=10,
      desc="Filtering to English-only",
  )
  print(f"Language filter: kept {len(raw):,}/{total_pre_lang:,} English examples "
        f"(removed {total_pre_lang - len(raw):,} non-English)")

  # Skip short-answer classification task types (no rationale, just labels)
  if skip_task_types and "task_type" in raw.column_names:
    skip_set = set(skip_task_types)
    total_pre_skip = len(raw)
    raw = raw.filter(
        lambda ex: ex["task_type"] not in skip_set,
        num_proc=10,
        desc="Filtering out classification task types",
    )
    print(f"Task-type filter: kept {len(raw):,}/{total_pre_skip:,} "
          f"(removed {total_pre_skip - len(raw):,} from {skip_set})")

  source_key = LAWINSTRUCT_DICT["source"]  # "dataset_name" by default
  counts_before = Counter(raw[source_key])
  total_before = len(raw)

  # --- Stats ---
  print(f"\n{'='*70}")
  print(f"LawInstruct stats  (config={lawinstruct_config})")
  print(f"{'='*70}")
  print(f"Total examples (after empty-answer filter): {total_before:,}")
  print(f"Unique sub-datasets (by '{source_key}'): {len(counts_before)}")
  if "task_type" in raw.column_names:
    tt_counts = Counter(raw["task_type"])
    print(f"Task types ({len(tt_counts)}):")
    for tt, c in sorted(tt_counts.items(), key=lambda x: -x[1]):
      print(f"  {tt:40s} {c:>8,}")
  print(f"\nPer sub-dataset counts (before capping):")
  for ds_name, c in sorted(counts_before.items(), key=lambda x: -x[1]):
    print(f"  {ds_name:50s} {c:>8,}")

  if lawinstruct_cap_per_ds is not None:
    # Resolve per-group caps: < 1 means fraction, >= 1 means absolute
    cap_col = raw[lawinstruct_cap_key]
    group_counts = Counter(cap_col)

    if lawinstruct_cap_per_ds < 1:
      group_caps = {g: max(1, int(cnt * lawinstruct_cap_per_ds))
                    for g, cnt in group_counts.items()}
      print(f"\nCapping by '{lawinstruct_cap_key}' at {lawinstruct_cap_per_ds:.1%} per group:")
    else:
      group_caps = {g: int(lawinstruct_cap_per_ds) for g in group_counts}
      print(f"\nCapping by '{lawinstruct_cap_key}' at {int(lawinstruct_cap_per_ds)} per group:")

    for g, cap in sorted(group_caps.items(), key=lambda x: -group_counts[x[0]]):
      flag = " [CAPPED]" if group_counts[g] > cap else ""
      print(f"  {g:50s} {group_counts[g]:>8,} -> {cap:>8,}{flag}")

    # Vectorized capping using numpy for speed on large datasets
    cap_arr = np.array(cap_col)
    unique_groups_arr = np.unique(cap_arr)
    keep_mask = np.zeros(len(cap_arr), dtype=bool)
    for grp in unique_groups_arr:
      grp_indices = np.where(cap_arr == grp)[0]
      keep_mask[grp_indices[:group_caps[grp]]] = True
    raw = raw.select(np.where(keep_mask)[0].tolist())

    num_capped = sum(1 for g, cnt in group_counts.items() if cnt > group_caps[g])
    print(f"\nAfter capping: {len(raw):,} examples "
          f"({num_capped}/{len(group_counts)} groups were capped)")
  print(f"{'='*70}\n")

  train_ds, src2id, num_train_sources = process_data(raw, LAWINSTRUCT_DICT)
  return train_ds, src2id, num_train_sources


# ── Dolma (raw-text chunking for pretraining) ────────────────────────────────

# Prefix patterns for the 4 English domains we care about.
DOLMA_SOURCE_PREFIXES = {
    "cc_en":   ["cc_en", "common crawl", "c4"],
    "wiki":    ["wiki"],
    "stack":   ["stack"],
    "reddit":  ["reddit"],
}


def _match_dolma_source(source_str: str):
    """Map a Dolma `source` field value to one of our domain keys."""
    s = source_str.lower()
    for key, prefixes in DOLMA_SOURCE_PREFIXES.items():
        for p in prefixes:
            if p in s:
                return key
    return None


def get_dolma(
    tokenizer,
    max_seq_length: int,
    max_tokens: int = 1_000_000_000,
    seed: int = 42,
    dolma_config: str = "v1_6-sample",
):
    """Download Dolma, filter to English domains, tokenize, chunk into fixed-length sequences.

    Args:
        tokenizer: HuggingFace tokenizer (fast).
        max_seq_length: chunk length in tokens.
        max_tokens: total token budget across all domains (natural proportions).
        seed: random seed (for shuffle).
        dolma_config: which Dolma version/sample to load.

    Returns:
        (dataset, src2id, num_sources)
        dataset is a HuggingFace Dataset with columns:
            input_ids  (list[int], length max_seq_length)
            labels     (list[int], same as input_ids)
            source     (int, domain id)
    """
    src2id = {k: i for i, k in enumerate(DOLMA_SOURCE_PREFIXES.keys())}
    num_sources = len(src2id)

    print(f"Loading Dolma ({dolma_config}) — full download, budget={max_tokens:,} tokens")
    print(f"Domains: {src2id}")

    raw = datasets.load_dataset(
        "allenai/dolma", dolma_config, split="train",
        trust_remote_code=True,
    )
    raw = raw.shuffle(seed=seed)

    # Filter to our 4 English domains
    def _keep(ex):
        return _match_dolma_source(ex.get("source", "")) is not None
    raw = raw.filter(_keep, num_proc=10, desc="Filtering to target domains")
    print(f"After domain filter: {len(raw):,} documents")

    all_chunks = []
    total_tokens = 0

    pbar = tqdm(total=max_tokens, unit="tok", desc="Dolma chunking")
    for example in tqdm(raw, desc="Dolma docs", leave=False):
        if total_tokens >= max_tokens:
            break

        domain = _match_dolma_source(example.get("source", ""))
        token_ids = tokenizer(
            example["text"], add_special_tokens=False, truncation=False,
        )["input_ids"]

        # chunk into max_seq_length pieces, drop remainder
        for i in range(0, len(token_ids) - max_seq_length + 1, max_seq_length):
            chunk = token_ids[i : i + max_seq_length]
            all_chunks.append({
                "input_ids": chunk,
                "labels": chunk,  # LM objective: predict every token
                "source": src2id[domain],
            })
            total_tokens += max_seq_length
            pbar.update(max_seq_length)
            if total_tokens >= max_tokens:
                break

    pbar.close()

    print(f"Dolma done: {len(all_chunks):,} chunks, {total_tokens:,} tokens total")
    per_domain = Counter(c["source"] for c in all_chunks)
    id2src = {v: k for k, v in src2id.items()}
    for sid, cnt in sorted(per_domain.items()):
        print(f"  {id2src[sid]:10s}: {cnt:>8,} chunks")

    result = Dataset.from_list(all_chunks)
    result = result.shuffle(seed=seed)
    result.set_format(type="pt")
    return result, src2id, num_sources
