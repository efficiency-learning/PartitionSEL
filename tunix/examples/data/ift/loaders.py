import jax.numpy as jnp
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DataCollatorForSeq2Seq

from .source_masks import get_source_masks
from .utils import to_np_int32


def make_jax_collate(tok, num_train_sources, max_target_length, subsel_bs=None, include_full=None, include_sourcemasks=False):
    collator = DataCollatorForSeq2Seq(
      tokenizer=tok,
      model=None,
      padding="max_length",
      max_length=max_target_length,
      label_pad_token_id=-100,
      return_tensors="np",
    )

    def jax_collate(features):
      ret = {}
      if "source" in features[0].keys():
        sources = [to_np_int32(ex["source"]) for ex in features]
        if include_sourcemasks:
          source_meta = get_source_masks(sources, subsel_bs, num_train_sources, include_full=include_full)
        else:
          source_meta = {"sources": jnp.array(sources, dtype=jnp.int32)}
        ret["meta"] = source_meta

      batch = collator(features)
      input_ids = jnp.asarray(batch["input_ids"], dtype=jnp.int32)
      labels = jnp.asarray(batch["labels"], dtype=jnp.int32)
      input_mask = (labels != -100).astype(jnp.int32)
      ret = {"input_tokens": input_ids, "input_mask": input_mask, **ret}
      return ret

    return jax_collate


class InfiniteLoader:
  def __init__(self, loader):
    self.loader = loader
    self.iterator = iter(loader)

  def __iter__(self):
    return self

  def __next__(self):
    try:
      return next(self.iterator)
    except StopIteration:
      self.iterator = iter(self.loader)  # new epoch
      return next(self.iterator)

  def __len__(self):
    # number of batches per epoch
    return len(self.loader)


def make_seeded_loader(ds, batch_size, collate_fn, seed=42, shuffle=True,
                       num_workers=0, infinite=False,
                       domain_weights=None, src2id=None):
  g = torch.Generator().manual_seed(seed)

  sampler = None
  if domain_weights is not None and len(domain_weights) > 0:
    import json
    from tqdm import tqdm
    with open(domain_weights, "r") as f: weight_map = json.load(f)["train_domain_weights"]
    weight_map = {src2id[k]: weight_map[k] for k in weight_map.keys()}
    assert weight_map is not None
    assert src2id is not None
    print("Doing weighted sampling with", weight_map)
    weights = []
    for i in tqdm(range(len(ds)), desc="Mapping Weights"):
      src = ds[i]["source"].item()
      assert src is not None
      wt = weight_map[src]
      weights.append(wt)
    weights = torch.tensor(weights, dtype=torch.float)

    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=g)
    shuffle = False  # sampler and shuffle are mutually exclusive

  loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler,
                      shuffle=shuffle, generator=g, num_workers=num_workers, drop_last=True)

  return InfiniteLoader(loader) if infinite else loader
