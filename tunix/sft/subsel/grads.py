from __future__ import annotations

from typing import Any, Tuple, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from functools import partial
import optax
from tunix.sft import utils
from tunix.sft.subsel.utils import *
from tunix.sft.subsel.dimred import *
import chex


def post_process_grad(grads, task_cfg):
  normalize_flag = task_cfg["grads"]["normalize"]
  if normalize_flag:
    grads = normalize(grads)
  return grads

def update_moments(ex_grads, mu, nu, count):
  # if not lowpass_adam: return ex_grads, None, None
  # if mu is None: return ex_grads, None, None
  eps=1e-8
  b1=0.9
  b2=0.999 

  def bias_correct(mu, nu, count):
    count_inc = count + 1
    mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
    nu_hat = optax.tree.bias_correction(nu, b2, count_inc)
    return mu_hat, nu_hat
  
  def get_mu_nu(grad, mu, nu):
    mu = optax.tree.update_moment(grad, mu, b1, 1)
    nu = optax.tree.update_moment_per_elem_norm(grad, nu, b2, 2)
    return mu, nu
    
  # jax.debug.print("grads shsape {}", jax.tree.map(lambda x: x.shape, ex_grads))
  def scale(ex_grad):
    _mu, _nu = get_mu_nu(ex_grad, mu, nu)
    _mu_hat, _nu_hat = bias_correct(_mu, _nu, count)
    smooth = jax.tree.map(
      lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps), _mu_hat, _nu_hat, is_leaf=lambda x: x is None,
    )

    return smooth, _mu, _nu
  
  grads, mu_batch, nu_batch = jax.vmap(scale)(ex_grads)
  # grads = ex_grads

  return grads, mu_batch, nu_batch

def chunk_fn(model, inputs, dimred_dim, grad_layer, rng, chunk_size=16, use_lora=None):

  def stack(trees, axis=0): return jax.tree.map(lambda *xs: jnp.concat(xs, axis=axis), *trees)
  size = jax.tree.leaves(inputs)[0].shape[0]
  chunks = max(size//chunk_size, 1)
  grads, hidden, delta, loss = [], [], [], []
  for i in range(chunks):
    start = i*chunk_size
    end = start + chunk_size
    sub = {}
    for k in inputs.keys():
      if k == "meta": continue
      sub[k] = inputs[k][start: end]
    _grad, _hidden, _delta, _loss = _chunk_per_ex_grads(model, grad_layer, rng, sub, use_lora=use_lora)

    # _grad = pack_pytree_layered(_grad)
    # _grad = jax.tree.map(lambda x: dimred_fft(rng, x, dimred_dim), _grad)
    # _grad = pack_pytree(_grad)

    grads.append(_grad)
    hidden.append(_hidden)
    delta.append(_delta)
    loss.append(_loss)

  return stack(grads), stack(hidden), stack(delta), stack(loss)


def get_grad_shape(
    model, inputs: Any, rng, chunk_size=16,
) -> ArrayLike | Tuple[ArrayLike, Any]: 
  dimred_dim = task_cfg["grads"]["dimred_dim"]
  grad_layer = task_cfg["grads"]["grad_layer"]
  grads, *_ = chunk_fn(model, inputs, dimred_dim, grad_layer, rng,chunk_size=chunk_size)
  grads = jax.tree.map(lambda x: jnp.zeros_like(x), grads)
  return grads


def per_ex_grads(
    model, step, inputs: Any, task_cfg, grad_layer, rng, chunk_size=16, moments=None, lowpass=False,
) -> ArrayLike | Tuple[ArrayLike, Any]: 
  # if "meta" in inputs.keys():
  #   inputs.pop("meta")
  dimred_dim = task_cfg["grads"]["dimred_dim"]
  dimred = task_cfg["grads"]["dimred"]
  use_lora = task_cfg["grads"]["use_lora"]

  grads, hidden, delta, loss = chunk_fn(model, inputs, dimred_dim, grad_layer, rng, chunk_size=chunk_size, use_lora=use_lora)
  # dimred_dim = model.config.embed_dim
  meta = {
    "hidden": hidden,
    "delta": delta,
    "input_loss": loss.mean(),
    "moments": None
  }

  if lowpass:
    grads, mu_hat, nu_hat = update_moments(grads, moments[0], moments[1], step)
    meta["moments"] = (mu_hat, nu_hat)
    grads = pack_pytree(grads)

  if dimred :
    # FIXME: assume lowpass and dimred cant be both true at same time for now
    assert not (dimred and lowpass)
    grads = pack_pytree_layered(grads)
    grads = jax.tree.map(lambda x: dimred_fft(rng, x, dimred_dim), grads)
    grads = pack_pytree(grads)

  return grads, meta


def grad_filter(grad_layer, use_lora, path, value):
  # jax.debug.print("path {}",path)
  # ('layers', Array(1, dtype=int32), 'mlp', 'down_proj', 'kernel_lora_a')
  if use_lora:
    # LoRA mode: differentiate through LoRA params only
    if grad_layer is None:
      return nnx.LoRAParam

    if len(path) > 1 and isinstance(path[1], (int, jax.Array)):
      idx = int(path[1]) if isinstance(path[1], jax.Array) else path[1]
      if idx in grad_layer:
        for cand in ["w_lora_a", "w_lora_b", "kernel_lora_a", "kernel_lora_b"]:
           if cand in path[-1]:
            return True
    return False
  else:
    # Full-weight mode (pretraining): differentiate through regular params
    if grad_layer is None:
      return nnx.Param

    if len(path) > 1 and isinstance(path[1], (int, jax.Array)):
      idx = int(path[1]) if isinstance(path[1], jax.Array) else path[1]
      if idx in grad_layer:
        # match typical weight names (kernel, w, embedding, etc.)
        return isinstance(value, nnx.Param) and not isinstance(value, nnx.LoRAParam)
    return False

def _chunk_per_ex_grads(
    model, grad_layer, rng, inputs: Any, use_lora=None
) -> ArrayLike | Tuple[ArrayLike, Any]:
  input_tokens = inputs["input_tokens"]
  input_mask = inputs["input_mask"]
  positions = inputs["positions"]
  attention_mask = inputs["attention_mask"]

  grad_fn = nnx.value_and_grad(
    single_ex_loss,
    argnums=nnx.DiffState(0, partial(grad_filter, grad_layer, use_lora)),
    has_aux=True
  )
  grad_fn = nnx.vmap(grad_fn, in_axes=(None, 0, 0, 0, 0))

  loss, grads = grad_fn(model, input_tokens,input_mask,positions,attention_mask)
  loss, (hidden, delta) = loss
  
  # grads = jax.tree.map(lambda x: x.astype(jnp.bfloat16), grads)
  # grads = jax.tree.map(lambda x: dimred_fft(rng, x, dimred_dim), grads)
  grads = jax.tree.map(lambda x: x.reshape(x.shape[0], -1).astype(jnp.bfloat16), grads)



  # jax.debug.print("inside {}", jax.tree.map(lambda x: x.shape, grads))
  return grads, hidden, delta, loss


@partial(nnx.jit)
def single_ex_loss(
    model,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
  """Per-sequence masked cross-entropy. If batch_mean=False, returns [B] vector."""
  input_tokens = input_tokens[None, :]
  input_mask = input_mask[None, :]
  positions = positions[None, :]
  attention_mask = attention_mask[None, :]
  per_seq_mean, (hidden, delta) = per_ex_loss(model, input_tokens, input_mask, positions, attention_mask)

  return per_seq_mean.sum(), (hidden[0], delta[0])


# @partial(nnx.jit, donate_argnames=("model", "input_tokens", "input_mask", "positions", "attention_mask"))
def per_ex_loss(
    model,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
  logits, _ = model(input_tokens, positions, None, attention_mask, output_hidden_states=True)  # logits [B,T,V], hidden [B,T,D]
  hidden = nnx.pop(model, nnx.Intermediate)[
        'all_hidden_states'
    ].value
  # jax.debug.print("hidden len {}", len(hidden))
  # jax.debug.print("hidden shape {}", hidden[-1].shape)
  # hidden = hidden[-1]
  hidden = jax.tree.map(lambda x: x.astype(jnp.bfloat16), hidden)
  hidden = jnp.stack(hidden, axis=0) # [N,B,T,D]
  hidden = jnp.transpose(hidden, (1, 0, 2, 3)) # [B,N,T,D]
  
  logits = logits.astype(jnp.float32)

  logits = logits[:, :-1, :]                 # [B,T-1,V]
  target_tokens = input_tokens[:, 1:]        # [B,T-1]
  target_mask  = input_mask[:, 1:]           # [B,T-1]

  log_probs = jax.nn.log_softmax(logits, axis=-1)                      # [B,T-1,V]
  one_hot   = jax.nn.one_hot(target_tokens, logits.shape[-1], dtype=jnp.float32)  # [B,T-1,V]

  nll_tok = -jnp.sum(log_probs * one_hot, axis=-1) * target_mask       # [B,T-1]
  per_seq_sum   = jnp.sum(nll_tok, axis=-1)                            # [B]
  per_seq_count = jnp.maximum(jnp.sum(target_mask, axis=-1), 1e-8)     # [B]
  per_seq_mean  = per_seq_sum / per_seq_count                          # [B]

  # ----- delta: per-token gradient w.r.t. final hidden states -----
  probs = jnp.exp(log_probs)                                           # [B,T-1,V]
  delta = (probs - one_hot) * target_mask[:, :, None]                   # [B,T-1,V]

  # Map vocab-space gradient to hidden-space via vocab head weights.
  # If your head computes logits = hidden @ W (W is [D,V]), this is correct:
  if hasattr(model.config, "use_tied_embedding") and model.config.use_tied_embedding:
    W = model.embedder.input_embedding.value.T
  elif hasattr(model.config, "weight_tying") and model.config.weight_tying:
    W = model.embedder.input_embedding.value.T
  else:
    W = model.lm_head.w.value # [D,V]
  # W = model.lm_head.w.value # [D,V]
    
  delta = jnp.einsum("BTV,DV->BTD", delta, W).astype(jnp.bfloat16)    # [B,T-1,D]
  # Pad to [B,T,D] to align with hidden's T dimension (last position has no target)
  delta = jnp.pad(delta, ((0,0),(0,1),(0,0)))                           # [B,T,D]

  return per_seq_mean, (hidden, delta)

