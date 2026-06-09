from collections.abc import Iterable

import datasets

import tunix.sft.eval.chat_templates as chat_templates
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft.eval.data_selection import get_training_dataset, get_validation_dataset
from tunix.sft.peft_trainer import TrainingInput  # pylint: disable=g-importing-member

from .configs import LAWINSTRUCT_DICT, METAMATH_DICT, MOL_DICT, TIGER_DICT
from .datasets import (
    get_colm,
    get_dolma,
    get_eval_greats,
    get_lawinstruct,
    get_legalbench,
    process_data,
)
from .loaders import make_jax_collate, make_seeded_loader
from .utils import debug, filter_all_ignored_labels, get_subjects, split_data


def create_datasets(
  dataset_name: str,
  global_batch_size: int,
  eval_global_batch_size: int,
  max_target_length: int,
  num_train_epochs: int | None,
  tokenizer: tokenizer_lib.Tokenizer,
  *,
  split_ratio: float = 0.005,
  cache_dir = None,
  answer_only_mask: bool = True,
  subsel_bs=None,
  config = None
) -> tuple[Iterable[TrainingInput], Iterable[TrainingInput]]:

  if dataset_name in ["TIGER-Lab/MathInstruct", "meta-math/MetaMathQA", "zjunlp/Mol-Instructions", "lawinstruct/lawinstruct"]:
    print(f"@@@@@@@@ Training on {dataset_name}")
    tok = tokenizer._tokenizer
    tok.chat_template = chat_templates.qwen2_5_template
    task_cfg = config["task_config"]["config"]
    include_full = None
    if task_cfg["subsel"]["minority_full"]: include_full = task_cfg["subsel"]["minority_classes"]
    domain_weights=None
    if task_cfg["domain_weights"]["enabled"]:
      domain_weights=task_cfg["domain_weights"]["weights_dir"]
    if dataset_name == "TIGER-Lab/MathInstruct":
      raw = datasets.load_dataset(dataset_name, split="train")
      raw = raw.shuffle(seed=42)
      raw, src2id, num_train_sources = process_data(raw, TIGER_DICT)
      train_ds = raw
      train_on_input = False
      
    elif dataset_name == "meta-math/MetaMathQA":
      raw = datasets.load_dataset(dataset_name, split="train")
      raw = raw.shuffle(seed=42)
      raw = raw.train_test_split(test_size=10000, seed=42, shuffle=False)
      raw, _ = raw["train"], raw["test"]
      raw, src2id, num_train_sources = process_data(raw, METAMATH_DICT)
      train_ds = raw
      train_on_input = False
      

    elif dataset_name == "zjunlp/Mol-Instructions":
      cfgs = ['Molecule-oriented Instructions', 'Protein-oriented Instructions', 'Biomolecular Text Instructions']
      keys = [
        "description_guided_molecule_design",
        "forward_reaction_prediction",
        "reagent_prediction",
        "retrosynthesis"
      ]
      # Hold out test_size samples per domain; train on the rest (natural imbalance).
      # Override via task_config.config.test_holdout_per_domain (default: 1000).
      test_holdout = task_cfg.get("test_holdout_per_domain", 1000)
      print(f"Mol-Instructions test_holdout_per_domain: {test_holdout}")
      raw = datasets.load_dataset("zjunlp/Mol-Instructions", cfgs[0], trust_remote_code=True)
      # extract test set per group add source col and concat
      train_ds = datasets.concatenate_datasets([
        (split := raw[k].train_test_split(test_size=test_holdout, seed=42, shuffle=True))['train']
            .add_column("source", [k] * len(split['train']))
        for k in keys
      ])
      train_ds = train_ds.shuffle(seed=42)
      train_ds, src2id, num_train_sources = process_data(train_ds, MOL_DICT)
      
      # sauce: https://github.com/zjunlp/Mol-Instructions/blob/main/demo/finetune.py#L49C9-L49C24
      train_on_input = True

    elif dataset_name == "lawinstruct/lawinstruct":
      # All task_types in lawinstruct (all_english-1_english):
      #   NATURAL_LANGUAGE_INFERENCE  1,156,490
      #   QUESTION_ANSWERING           525,515
      #   TEXT_CLASSIFICATION           379,776
      #   NAMED_ENTITY_RECOGNITION      84,781
      #   SUMMARIZATION                  42,917
      #   MULTIPLE_CHOICE                10,893
      #   ARGUMENTATION                   3,456
      #   QUESTION_GENERATION             2,442
      train_ds, src2id, num_train_sources = get_lawinstruct(
          hf_config="all_english-1_english",
          cap_per_group=None,
          cap_key="task_type",
          skip_task_types={
              "TEXT_CLASSIFICATION",
              "MULTIPLE_CHOICE",
              "NAMED_ENTITY_RECOGNITION",
              "NATURAL_LANGUAGE_INFERENCE",
          },
          
      )
      # Don't train on the instruction/prompt — only on the answer
      train_on_input = True

    else:
      raise NotImplementedError

    eval_source = task_cfg["subsel"]["eval_source"]
    dev_source = task_cfg["subsel"]["dev_source"]
    eval_mask_input = task_cfg["subsel"].get("eval_mask_input", False)
    dev_mask_input = task_cfg["subsel"].get("dev_mask_input", False)
    print("EVAL SOURCE", eval_source)
    print("DEV SOURCE", dev_source)
    print(f"eval_mask_input={eval_mask_input}, dev_mask_input={dev_mask_input}")
    print("train on input", train_on_input)

    # TODO: split ratio things
    if eval_source == "train":
      print("Using random split from train as eval")
      train_ds, eval_ds = split_data(train_ds, group_column=None,
                                       test_size=split_ratio, seed=42, shuffle=True)
      eval_ds = get_training_dataset.encode_datav2(
          eval_ds, tok, max_target_length, mask_input=eval_mask_input)

    elif eval_source == "colm":
      print("Using COLM as eval")
      eval_ds = get_colm(tok, 5, max_target_length, mask_input=eval_mask_input)

    elif eval_source == "legalbench":
      print(f"Using LegalBench as eval (held-out from train split, mask_input={eval_mask_input})")
      eval_ds = get_legalbench(tok, max_target_length, mask_input=eval_mask_input)


    if dev_source == "train":
      print("Using random split from train as dev")
      train_ds, dev_ds = split_data(train_ds, group_column="source",
                                       test_size=split_ratio, seed=42, shuffle=True)
      dev_ds = get_training_dataset.encode_datav2(
          dev_ds, tok, max_target_length, mask_input=dev_mask_input)

    elif dev_source == "eval":
      print("Using eval as dev")
      dev_ds = eval_ds

    elif dev_source == "colm":
      print("Using COLM as dev")
      dev_ds = get_colm(tok, 5, max_target_length, mask_input=dev_mask_input)

    elif dev_source == "legalbench":
      print(f"Using LegalBench as dev (mask_input={dev_mask_input})")
      if eval_source == "legalbench":
        # Both eval+dev from LegalBench: split to avoid overlap
        _split = eval_ds.train_test_split(test_size=0.2, seed=42, shuffle=True)
        eval_ds = _split["train"]
        dev_ds = _split["test"]
      else:
        dev_ds = get_legalbench(tok, max_target_length, mask_input=dev_mask_input)

    train_ds = get_training_dataset.encode_datav2(
        train_ds, tok, max_target_length, mask_input=not train_on_input, verbose=True)
    
    jax_collate = make_jax_collate(tok, num_train_sources, max_target_length, subsel_bs=subsel_bs,
                                   include_full=include_full, include_sourcemasks=True)
    jax_collate_dev = make_jax_collate(tok, num_train_sources, max_target_length, subsel_bs=subsel_bs,
                                    include_sourcemasks=False)
    train_ds = filter_all_ignored_labels(train_ds, num_proc=10, desc="Filtering Train",
                                         debug=True, tokenizer=tok, debug_n=5)
    train_ds = make_seeded_loader(train_ds, batch_size=global_batch_size, 
                                  domain_weights=domain_weights,
                                  collate_fn=jax_collate, seed=42, infinite=True, src2id=src2id)

    eval_ds = filter_all_ignored_labels(eval_ds, num_proc=10, desc="Filtering Val",
                                         debug=True, tokenizer=tok, debug_n=5)
    eval_ds = make_seeded_loader(eval_ds, batch_size=eval_global_batch_size, 
                                  collate_fn=jax_collate_dev, seed=42, src2id=src2id)
    
    dev_ds = filter_all_ignored_labels(dev_ds, num_proc=10, desc="Filtering Dev",
                                         debug=True, tokenizer=tok, debug_n=5)
    dev_ds = make_seeded_loader(dev_ds, batch_size=eval_global_batch_size, 
                                  collate_fn=jax_collate_dev, seed=42, src2id=src2id)

    return train_ds, eval_ds, dev_ds, {"num_train_sources": num_train_sources}

  elif dataset_name == "allenai/dolma":
    print(f"@@@@@@@@ Training on {dataset_name} (raw-text chunking)")
    tok = tokenizer._tokenizer
    task_cfg = config["task_config"]["config"]

    dolma_cfg = task_cfg.get("dolma", {})
    max_tokens = dolma_cfg.get("max_tokens", 1_000_000_000)  # default 1B tokens
    dolma_version = dolma_cfg.get("version", "v1_6-sample")

    # Load, tokenize, chunk
    train_ds, src2id, num_train_sources = get_dolma(
        tok, max_target_length, max_tokens=max_tokens,
        seed=42, dolma_config=dolma_version,
    )

    # Split off eval + dev from train (already tokenized)
    train_ds, eval_ds = split_data(train_ds, group_column=None,
                                   test_size=split_ratio, seed=42, shuffle=True)
    train_ds, dev_ds = split_data(train_ds, group_column="source",
                                  test_size=split_ratio, seed=42, shuffle=True)

    jax_collate = make_jax_collate(tok, num_train_sources, max_target_length,
                                   subsel_bs=subsel_bs,
                                   include_sourcemasks=True)
    jax_collate_dev = make_jax_collate(tok, num_train_sources, max_target_length,
                                       subsel_bs=subsel_bs, include_sourcemasks=False)

    train_ds = filter_all_ignored_labels(train_ds, num_proc=10, desc="Filtering Train",
                                         debug=True, tokenizer=tok, debug_n=5)
    train_ds = make_seeded_loader(train_ds, batch_size=global_batch_size,
                                  collate_fn=jax_collate, seed=42, infinite=True, src2id=src2id)

    eval_ds = filter_all_ignored_labels(eval_ds, num_proc=10, desc="Filtering Val",
                                         debug=True, tokenizer=tok, debug_n=5)
    eval_ds = make_seeded_loader(eval_ds, batch_size=eval_global_batch_size,
                                  collate_fn=jax_collate_dev, seed=42, src2id=src2id)

    dev_ds = filter_all_ignored_labels(dev_ds, num_proc=10, desc="Filtering Dev",
                                        debug=True, tokenizer=tok, debug_n=5)
    dev_ds = make_seeded_loader(dev_ds, batch_size=eval_global_batch_size,
                                 collate_fn=jax_collate_dev, seed=42, src2id=src2id)

    return train_ds, eval_ds, dev_ds, {"num_train_sources": num_train_sources}

  elif dataset_name == "greats":
    data_dir = [
      "/home/aiscuser/prayas/temp/data/train/processed/cot/cot_data.jsonl",
      "/home/aiscuser/prayas/temp/data/train/processed/dolly/dolly_data.jsonl",
      "/home/aiscuser/prayas/temp/data/train/processed/flan_v2/flan_v2_data.jsonl",
      "/home/aiscuser/prayas/temp/data/train/processed/oasst1/oasst1_data.jsonl",
    ]
    
    print("###########TRAIN#############")
    debug("/home/aiscuser/prayas/temp/data/train")
    print("###########EVAL#############")
    debug("/home/aiscuser/prayas/temp/data/eval/mmlu/dev")

    ddir = "/home/aiscuser/prayas/temp/data"
    tok = tokenizer._tokenizer
    tok.chat_template = chat_templates.qwen2_5_template
    jax_collate = make_jax_collate(tok, max_target_length)

    train_ds = get_training_dataset.get_training_dataset(data_dir, tok, max_target_length, seed=42)
    print("done 1")
    train_ds = filter_all_ignored_labels(train_ds, num_proc=10)
    train_ds = make_seeded_loader(train_ds, batch_size=global_batch_size, collate_fn=jax_collate, seed=42)

    n_val = 2000
    task = "mmlu"
    subject = get_subjects("/home/aiscuser/prayas/temp/data/eval/mmlu/dev")
    print("Subjects", subject)
    wordy_dev = config["task_config"]["config"]["wordydev"]
    eval_ds = get_validation_dataset.get_dataset(
        task,
        data_dir=ddir,
        tokenizer=tok,
        max_length=max_target_length,
        validation=True,
        k=n_val,
        subject=subject,
        append_choice_text="eval" in wordy_dev
    )
    eval_ds.set_format(type="pt")
    eval_ds = filter_all_ignored_labels(eval_ds, num_proc=10)
    eval_ds = make_seeded_loader(eval_ds, batch_size=eval_global_batch_size, collate_fn=jax_collate, seed=42)
    dev_ds = get_validation_dataset.get_dataset(
        task,
        data_dir=ddir,
        tokenizer=tok,
        max_length=max_target_length,
        validation=True,
        k=n_val,
        subject=subject,
        append_choice_text="dev" in wordy_dev
    )
    dev_ds.set_format(type="pt")
    dev_ds = filter_all_ignored_labels(dev_ds, num_proc=10)
    dev_ds = make_seeded_loader(dev_ds, batch_size=eval_global_batch_size, collate_fn=jax_collate, seed=42)

    return train_ds, eval_ds, dev_ds
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")
