import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike  # pylint: disable=g-importing-member


def default_loss_fn(
    model: nnx.Module,
    inputs,
) -> ArrayLike:
  """Default loss function for PEFT training."""
  input_tokens = inputs["input_tokens"]
  input_mask = inputs["input_mask"]
  positions = inputs["positions"]
  attention_mask = inputs["attention_mask"]

  logits, _ = model(input_tokens, positions, None, attention_mask)
  logits = logits.astype(jnp.float32)
  # Exclude the last step as it does not appear in the targets.
  logits = logits[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Convert the target labels to one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

  # Define the normalization factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the negative log likelihood (NLL) loss.
  # Equivalent to: optax.softmax_cross_entropy(logits, one_hot).mean()
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor, None


def loss_unpacked(
    model,
    inputs,
    aux
) -> ArrayLike:
  """Per-sequence masked cross-entropy. If batch_mean=False, returns [B] vector."""
  input_mask = inputs["input_mask"]
  input_tokens = inputs["input_tokens"]
  positions = inputs["positions"]
  attention_mask = inputs["attention_mask"]

  logits, _, = model(input_tokens, positions, None, attention_mask)  # [B,T,V]
  logits = logits.astype(jnp.float32)
  logits = logits[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask  = input_mask[:, 1:]  # [B,T-1]

  log_probs = jax.nn.log_softmax(logits, axis=-1)                    # [B,T-1,V]
  one_hot   = jax.nn.one_hot(target_tokens, logits.shape[-1])        # [B,T-1,V]
  nll_tok   = -jnp.sum(log_probs * one_hot, axis=-1) * target_mask   # [B,T-1]

  per_seq_sum   = jnp.sum(nll_tok, axis=-1)                          # [B]
  per_seq_count = jnp.maximum(jnp.sum(target_mask, axis=-1), 1e-8)    # [B]
  per_seq_mean  = per_seq_sum / per_seq_count                        # [B]
  aux = {}
  return per_seq_mean.mean(), aux
