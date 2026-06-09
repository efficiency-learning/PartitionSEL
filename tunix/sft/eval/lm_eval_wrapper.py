"""Wrapper to plug tunix JAX models into lm-evaluation-harness.

Key constraint from the tunix Sampler:
  The Sampler allocates a fixed KV cache of size `cache_size`.
  For any call: prompt_tokens + max_generation_steps <= cache_size.
  So we set max_length == cache_size, and everywhere we call the sampler
  we ensure max_prompt_length + max_generation_steps <= max_length.

Variables cheat-sheet (used throughout this file):
  max_length      = cache_size. Hard ceiling on total tokens (prompt + gen).
  max_gen_toks    = max generation steps. Budget reserved for output.
  max_prompt      = max_length - max_gen_toks. Budget left for the prompt.
                    Prompts exceeding this are truncated from the LEFT.
"""

import jax
import jax.numpy as jnp
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from tunix.generate import sampler as sampler_mod


class TunixLM(LM):
    """lm-eval-harness compatible wrapper around a tunix Sampler."""

    def __init__(
        self,
        tunix_sampler: sampler_mod.Sampler,
        tokenizer,
        batch_size: int = 16,
        max_length: int = 2048,
        max_gen_toks: int = 256,
    ):
        """Args:
          tunix_sampler: the Sampler instance (already has a KV cache allocated).
          tokenizer: HF tokenizer (not the tunix wrapper).
          batch_size: batch size for inference.
          max_length: MUST equal the Sampler's cache_size.
                      This is the hard ceiling: prompt_tokens + gen_steps <= max_length.
          max_gen_toks: max generation tokens. The effective prompt budget is
                        max_length - max_gen_toks.
        """
        super().__init__()
        self._sampler = tunix_sampler
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length            # == cache_size
        self._max_gen_toks = max_gen_toks

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "jax"

    def tok_encode(self, string: str):
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self._tokenizer.decode(tokens)

    def _pad_batch(self, items):
        """Pad batch to multiple of batch_size with dummy copies."""
        n = len(items)
        remainder = n % self._batch_size
        if remainder != 0:
            pad = self._batch_size - remainder
            items = items + items[:pad]
        return items, n

    # ------------------------------------------------------------------
    # generate_until
    # ------------------------------------------------------------------
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        contexts = []
        gen_kwargs_list = []
        for req in requests:
            ctx, gen_kwargs = req.args
            contexts.append(ctx)
            gen_kwargs_list.append(gen_kwargs)

        # max_gen: max tokens to GENERATE for this batch of requests.
        #          Each request can override via gen_kwargs["max_gen_toks"];
        #          we take the max across the batch so one sampler call covers all.
        max_gen = max(
            gk.get("max_gen_toks", self.max_gen_toks) for gk in gen_kwargs_list
        )
        # max_prompt: max prompt tokens we can fit = cache_size − gen budget.
        #             Prompts longer than this are left-truncated (keep the
        #             tail, which is the actual question being asked).
        # Sampler constraint: prompt_tokens + max_generation_steps <= cache_size (== max_length).
        max_prompt = self._max_length - max_gen

        # truncated: prompt strings after left-truncation to max_prompt tokens.
        truncated = []
        for ctx in contexts:
            ids = self.tok_encode(ctx)
            if len(ids) > max_prompt:
                ids = ids[-max_prompt:]  # keep the RIGHT (tail) end
            truncated.append(self.tok_decode(ids))

        # padded:  truncated list padded to a multiple of batch_size with dummy copies.
        # real_n:  how many elements in truncated are real (the rest are padding).
        padded, real_n = self._pad_batch(truncated)

        for i in range(0, len(padded), self._batch_size):
            batch = padded[i : i + self._batch_size]
            outs = self._sampler(
                batch,
                max_generation_steps=max_gen,
                max_prompt_length=max_prompt,  # NOT max_length; leaves room for gen steps
                echo=False,
                return_logits=False,
            )
            results.extend(outs.text)

        results = results[:real_n]

        # Truncate at stop sequences
        final = []
        for res_str, req in zip(results, requests):
            _, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            for stop in until:
                idx = res_str.find(stop)
                if idx != -1:
                    res_str = res_str[:idx]
            final.append(res_str)

        return final

    # ------------------------------------------------------------------
    # loglikelihood
    # ------------------------------------------------------------------
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []

        # Tokenize all requests.
        # ctx:      the context/prompt string (e.g. few-shot examples + question).
        # cont:     the continuation string whose likelihood we want to score.
        # ctx_ids:  token ids for the context.
        # cont_ids: token ids for the continuation.
        # full_ids: ctx_ids + cont_ids concatenated — the full token sequence.
        # cont_len: number of tokens in the continuation (we score only these).
        tokenized = []
        for req in requests:
            ctx, cont = req.args
            ctx_ids = self.tok_encode(ctx)
            cont_ids = self.tok_encode(cont)
            full_ids = ctx_ids + cont_ids
            # Left-truncate if full sequence exceeds cache.
            if len(full_ids) > self._max_length:
                full_ids = full_ids[-self._max_length :]
            cont_len = len(cont_ids)
            tokenized.append((full_ids, cont_len))

        # Process in batches — feed full string, get logits, score continuation
        for batch_start in range(0, len(tokenized), self._batch_size):
            batch_tok = tokenized[batch_start : batch_start + self._batch_size]

            # batch_strs: the full ctx+cont strings to feed the sampler.
            # cont_lens:  how many tokens at the END of each string are the
            #             continuation (we only compute logprobs over these).
            batch_strs = [self.tok_decode(ids) for ids, _ in batch_tok]
            cont_lens = [cl for _, cl in batch_tok]

            padded_strs, real_n = self._pad_batch(batch_strs)
            padded_cont_lens = cont_lens + cont_lens[: len(padded_strs) - real_n]

            # echo=True: sampler returns logits for the FULL input (prompt + 1 gen step).
            # max_prompt_length = max_length - 1 so that prompt + 1 gen step <= cache_size.
            outs = self._sampler(
                padded_strs,
                max_generation_steps=1,
                max_prompt_length=self._max_length - 1,  # leave 1 slot for the gen step
                echo=True,
                return_logits=True,
            )

            # outs.logits[j]: [seq_len, vocab] — logits at every echoed position.
            # outs.tokens[j]: [seq_len]        — token ids of the echoed input.
            for j in range(real_n):
                local_j = j % len(batch_strs)  # actual index in this sub-batch
                logits_j = outs.logits[j]      # [seq_len, vocab]
                tokens_j = outs.tokens[j]      # [seq_len]
                cont_len = padded_cont_lens[j] # how many tail tokens are the continuation

                if cont_len == 0:
                    results.append((0.0, True))
                    continue

                # log_probs: [seq_len, vocab] — log-softmax of the logits.
                # logits[t] predicts token[t+1], so to score the last cont_len
                # tokens we slice logits at [-(cont_len+1) : -1].
                log_probs = jax.nn.log_softmax(logits_j, axis=-1)

                # target_tokens: the actual continuation token ids we are scoring.
                target_tokens = tokens_j[-cont_len:]
                # pred_log_probs: the log-prob vectors that PREDICT each target token.
                #   pred_log_probs[k] predicts target_tokens[k].
                pred_log_probs = log_probs[-(cont_len + 1) : -1]

                # token_log_probs: scalar log-prob for each continuation token.
                # total_log_prob: sum of per-token log-probs = log P(continuation | context).
                token_log_probs = pred_log_probs[
                    jnp.arange(cont_len), target_tokens
                ]
                total_log_prob = float(jnp.sum(token_log_probs))

                # is_greedy: True if every continuation token is the argmax of the
                # model's distribution — i.e. greedy decoding would reproduce it.
                greedy_tokens = jnp.argmax(pred_log_probs, axis=-1)
                is_greedy = bool(jnp.all(greedy_tokens == target_tokens))

                results.append((total_log_prob, is_greedy))

        return results

    # ------------------------------------------------------------------
    # loglikelihood_rolling
    # ------------------------------------------------------------------
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        # Reuse loglikelihood with empty context
        ll_requests = []
        for req in requests:
            (string,) = req.args
            ll_requests.append(
                Instance(
                    request_type="loglikelihood",
                    args=("", string),
                    idx=req.idx,
                )
            )
        results = self.loglikelihood(ll_requests)
        return [logprob for logprob, _ in results]
