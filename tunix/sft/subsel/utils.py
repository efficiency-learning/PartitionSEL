from __future__ import annotations

from typing import Any, Tuple, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from functools import partial
import chex


def wt_mean(task_cfg, max_num_sources, sources, grads):
    if not task_cfg["subsel"]["val_srcwt"]: return grads
    class_ids = jnp.arange(max_num_sources, dtype=jnp.int32) # [MC]
    src_mask = (sources[None, :] == class_ids[:, None])       # [MC, N]
    num_src = jnp.sum(jnp.diff(jnp.sort(sources)) != 0) + 1
    counts = src_mask.sum(1) # [MC]
    wt_src = jnp.where(counts > 0.0, 1/counts, 0.0) # [MC]
    wt_src = wt_src/num_src
    wt_ex = wt_src[sources] # [N] 
    # anch [N, d]
    grads = (grads*wt_ex[:, None])
    grads_anchor = grads.sum(0, keepdims=True) # [1, d]
    chex.assert_tree_shape(grads_anchor, (1, grads.shape[1]))
    jax.debug.print("wt_ex {} {} {} sum={}", counts, num_src, wt_ex, wt_ex.sum())
    return grads_anchor

def pack_pytree(pytree):
  # tree leaves: [(B, D1), (B, D2), ...] -> concat: (B, Dtot)
  leaves, _ = jax.tree_util.tree_flatten(pytree)
  X = jnp.concatenate([leaf.reshape(leaf.shape[0], -1) for leaf in leaves], axis=1)          # (B, Dtot)
  return X


def pack_pytree_layered(state):
  # assumes `state` is dict-like with key "layers"
  return {i: pack_pytree(layer) for i, layer in state["layers"].items()}

# @partial(nnx.jit, static_argnames=("Dtot"))
def pad_features(x, Dtot):
    B, Dsmall = x.shape
    # if Dsmall == Dtot: return x
    assert Dsmall <= Dtot
    pad_width = ((0, 0), (0, Dtot - Dsmall))
    return jnp.pad(x, pad_width, mode="constant", constant_values=0)


def tree_sq_norm(pytree):
    # [B] per-sample squared L2 across all leaves
    leaves = jax.tree.leaves(pytree)
    # for i, x in enumerate(leaves):
    #     chex.assert_rank(x, {1, 2, 3, 4})  # must have batch dim
    n2 = jnp.zeros(leaves[0].shape[0], dtype=jnp.float32)
    for x in leaves:
        n2 = n2 + jnp.sum(jnp.square(x), axis=tuple(range(1, x.ndim)))
    # chex.assert_rank(n2, 1)  # must be [B]
    return n2

def normalize(pytree, eps=1e-8):
    # in: pytree{[B,...]} → out: pytree{[B,...]} with global L2=1 per sample
    leaves, treedef = jax.tree.flatten(pytree)
    # for i, x in enumerate(leaves):
    #     chex.assert_rank(x, {1, 2, 3, 4})
    n2 = jnp.zeros(leaves[0].shape[0], dtype=jnp.float32)
    for x in leaves:
        n2 = n2 + jnp.sum(jnp.square(x), axis=tuple(range(1, x.ndim)))
    # chex.assert_rank(n2, 1)
    inv = jax.lax.rsqrt(n2 + eps)                    # [B]
    normed = [x * inv.reshape((-1,) + (1,) * (x.ndim - 1)) for x in leaves]
    return jax.tree.unflatten(treedef, normed)

def gram_linear(X, Y=None):
  if Y is None: Y = X
  # return X@Y.T

  # sum_l X_l X_lᵀ → [B,B]
  def leaf_gram(x, y):
    x = x.astype(jnp.float32).reshape(x.shape[0], -1)              # [B,d]
    y = y.astype(jnp.float32).reshape(y.shape[0], -1)              # [B,d]
    chex.assert_rank(x, 2)
    chex.assert_rank(y, 2)
    # contracting dim d must be > 0
    chex.assert_axis_dimension_gt(x, 1, 0)
    chex.assert_axis_dimension_gt(y, 1, 0)
    return x @ y.T                         # [B,B]
  return jax.tree.reduce(lambda a, b: a + b, jax.tree.map(leaf_gram, X, Y))


def cross_entropy_loss(logits, targets, temp=1.0):
  
  targets = jax.nn.softmax(targets/temp, axis=-1)
  log_probs = jax.nn.log_softmax(logits/temp, axis=-1)
  return -jnp.sum(targets * log_probs, axis=-1).mean()

def mse_loss(logits, targets):
  loss = (logits - targets)**2
  return loss.mean()
  return loss