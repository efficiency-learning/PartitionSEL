"""Shared boilerplate for eval functions (COLM, MOL, LegalBench, etc.)."""

import os
from contextlib import contextmanager
from functools import partial

from tunix.generate import sampler
import tunix.sft.eval.chat_templates as chat_templates

RESULTS_ROOT = "/home/aiscuser/prayas/temp/.local/share/miniconda/sub-tunix/ret-subset/examples/sft/mtnt/results"


@contextmanager
def eval_setup(pipeline, model, mesh, tokenizer,
               max_prompt_tokens=1024, max_generation_steps=256,
               cache_size=2048, batch_size=16,
               eval_split="all", sub_dir=None):
    """Shared setup for eval functions.

    Yields (inf2, tok, out_dir) where:
      - inf2: callable, takes list[str] prompts -> list[str] completions
      - tok:  the raw HF tokenizer
      - out_dir: output directory for this eval run
    """
    exp_name = pipeline.config["training_config"]["checkpoint_root_directory"]
    ckpt_num = pipeline.config["inference_restore_step"]
    exp_name = os.path.basename(exp_name.rstrip("/"))
    tokenizer._tokenizer.chat_template = chat_templates.qwen2_5_template
    tok = tokenizer._tokenizer

    split_tag = f"_{eval_split}" if eval_split != "all" else ""
    out_dir = f"{RESULTS_ROOT}/{exp_name}/{ckpt_num}{split_tag}"
    if sub_dir:
        out_dir = os.path.join(out_dir, sub_dir)
    os.makedirs(out_dir, exist_ok=True)

    with mesh:
        mysamp = sampler.Sampler(
            model,
            tokenizer,
            sampler.CacheConfig(
                cache_size=cache_size,
                num_layers=model.config.num_layers,
                num_kv_heads=model.config.num_kv_heads,
                head_dim=model.config.head_dim,
            ),
        )
        inference_function = partial(
            pipeline.run_inference, mysamp,
            max_generation_steps, max_prompt_tokens, [], batch_size)
        inf2 = lambda batch: inference_function(batch)[0]

        yield inf2, tok, out_dir
