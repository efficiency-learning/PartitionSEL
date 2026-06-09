from pathlib import Path

import datasets
import numpy as np


def get_subjects(folder_path):
    names = []
    for p in Path(folder_path).iterdir():
        if p.is_file():
            name = p.stem
            if name.endswith("_dev"):
                name = name[:-4]  # remove '_dev'
            names.append(name)
    return names


def get_splits(ds, split_ratio):
  split_index = int((1 - split_ratio) * len(ds))
  print("SPLIT", split_index)
  train_ds = ds.select(range(0, split_index))
  val_ds = ds.select(range(split_index, len(ds)))
  return train_ds, val_ds


def debug(dir_path):
  total = 0
  for p in sorted(Path(dir_path).rglob("*")):
    if p.is_file() and p.suffix.lower() in {".jsonl", ".jsonls", ".csv"}:
      n = sum(1 for _ in p.open("rb"))
      print(f"{n:>10}  {p.name}")
      total += n
  print("-" * 40)
  print(f"{total:>10}  TOTAL")


def filter_all_ignored_labels(ds, *, num_proc=10, desc=None, debug=False,
                               tokenizer=None, debug_n=5):
  def has_valid_label(example):
    return (example["labels"] != -100).any().item()

  total = len(ds)

  # Identify which examples will be filtered out BEFORE filtering
  if debug and total > 0:
    keep_mask = [has_valid_label(ds[i]) for i in range(total)]
    dropped_idxs = [i for i, keep in enumerate(keep_mask) if not keep]
    n_show = min(debug_n, len(dropped_idxs))
    if n_show > 0:
      print(f"\n{'='*70}")
      print(f"DEBUG: {len(dropped_idxs)}/{total} examples being FILTERED OUT "
            f"(showing {n_show}):")
      print(f"{'='*70}")
      for rank, idx in enumerate(dropped_idxs[:n_show]):
        ex = ds[idx]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        n_tokens = len(input_ids)
        n_masked = (labels == -100).sum().item()
        print(f"\n--- Filtered example {rank+1} (idx={idx}) ---")
        print(f"  seq_len={n_tokens}, masked_labels={n_masked}/{n_tokens}")
        if tokenizer is not None:
          ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
          text = tokenizer.decode(ids, skip_special_tokens=False)
          # Show first 500 and last 200 chars to see both prompt and where response would be
          if len(text) > 800:
            print(f"  text (first 500): {text[:500]}")
            print(f"  text (last  200): ...{text[-200:]}")
          else:
            print(f"  text: {text}")
        else:
          ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
          print(f"  input_ids (first 20): {ids[:20]}")
          print(f"  input_ids (last  20): {ids[-20:]}")
        if "source" in ex:
          print(f"  source={ex['source']}")
      print(f"{'='*70}\n")

  ds = ds.filter(
      has_valid_label,
      num_proc=num_proc,
      desc=desc or "Filtering samples with all labels = -100",
  )
  skipped = total - len(ds)
  print(f"Skipped {skipped}/{total} ({skipped / total:.2%}) samples")
  return ds


def to_np_int32(x):
    import jax.numpy as jnp
    if hasattr(x, "numpy"):  # tensor
        return x.cpu().numpy().astype(jnp.int32)
    return jnp.asarray(x, dtype=jnp.int32)  # numpy or list


def split_by_group(ds, group_column, test_size=0.1, seed=42, shuffle=True):
    assert isinstance(ds, datasets.Dataset)
    unique_groups = set(ds[group_column])

    train_parts, test_parts = [], []

    for i, g in enumerate(unique_groups):
        subset = ds.filter(lambda x: x[group_column] == g)
        split = subset.train_test_split(
            test_size=test_size,
            seed=seed + i,
            shuffle=shuffle,
        )
        train_parts.append(split["train"])
        test_parts.append(split["test"])

    train_ds = datasets.concatenate_datasets(train_parts)
    test_ds = datasets.concatenate_datasets(test_parts)

    if shuffle:
        train_ds = train_ds.shuffle(seed=seed)
        test_ds = test_ds.shuffle(seed=seed + 1)

    return train_ds, test_ds


def repeat_dataset(ds, times, seed=42):
    out = []
    for i in range(times):
        out.append(ds.shuffle(seed + i))
    return datasets.concatenate_datasets(out)


def split_data(obj, group_column=None, test_size=0.1, seed=42, shuffle=True):
  if group_column is None:
    split = obj.train_test_split(test_size=test_size, seed=seed, shuffle=shuffle)
    return split["train"], split["test"]

  return split_by_group(obj, group_column, test_size, seed, shuffle)
