
from __future__ import annotations

from typing import Any, Tuple, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from functools import partial
from tunix.sft.subsel.utils import *


def dimred(grads, k):
  topk_tree = jax.tree.map(lambda x: jax.lax.top_k(x.mean(0), k)[1], grads)
  grads = jax.tree.map(lambda idxs, x: x[:, idxs], topk_tree, grads)
  return grads

import math

@partial(
  nnx.jit,
  static_argnames=("k", "deterministic")
)
def dimred_fft(rng, grads, k, deterministic=False):
  if deterministic:
    rng = jax.random.key(0)

  if grads.ndim == 2:
    # (bs, dim) → sketch along last axis → (bs, k)
    rng, key = jax.random.split(rng)
    n = grads.shape[-1]
    knew = min(k // 2, n // 2)
    signs, idx = make_fft_sketch_params(key, n, knew)
    return fft_sketch_real(grads, signs, idx)

  elif grads.ndim == 3:
    # (bs, a, b) → bilinear sketch per sample → (bs, k)
    bs, a, b = grads.shape
    # Budget: (2*knew_b) * (2*knew_a) ≈ k
    # Proportional split: knew_b/knew_a ≈ b/a
    knew_b = min(max(1, round(math.sqrt(k * b / (4 * a)))), b // 2)
    knew_a = min(max(1, round(math.sqrt(k * a / (4 * b)))), a // 2)

    # Sketch along axis=-1 (b): (bs, a, b) → (bs, a, 2*knew_b)
    rng, key1 = jax.random.split(rng)
    signs_b, idx_b = make_fft_sketch_params(key1, b, knew_b)
    sketched = fft_sketch_real(grads, signs_b, idx_b)       # (bs, a, 2*knew_b)

    # Sketch along axis=1 (a): swap axes, sketch, swap back
    # (bs, a, 2*knew_b) → transpose(0,2,1) → (bs, 2*knew_b, a) → sketch → (bs, 2*knew_b, 2*knew_a) → transpose back
    rng, key2 = jax.random.split(rng)
    signs_a, idx_a = make_fft_sketch_params(key2, a, knew_a)
    sketched = sketched.transpose(0, 2, 1)                  # (bs, 2*knew_b, a)
    sketched = fft_sketch_real(sketched, signs_a, idx_a)     # (bs, 2*knew_b, 2*knew_a)

    return sketched.reshape(bs, -1)                          # (bs, 4*knew_a*knew_b)

  else:
    raise ValueError(f"dimred_fft: expected 2D (bs,dim) or 3D (bs,a,b) input, got {grads.shape}D")

def make_fft_sketch_params(key, n, k):
  # Random Rademacher signs: ±1
  key1, key2 = jax.random.split(key)
  signs = jax.random.rademacher(key1, (n,), dtype=jnp.int8).astype(jnp.float32)

  # Choose k frequency bins uniformly (don’t bias to low freqs)
  F = n // 2 + 1
  idx = jax.random.choice(key2, F, shape=(k,), replace=False)
  idx = jnp.sort(idx)
  return signs, idx

def fft_sketch_real(grads, signs, idx):
    n = grads.shape[-1]
    x = grads * signs

    spec = jnp.fft.rfft(x, axis=-1, norm="ortho")   # [..., F] complex
    picked = jnp.take(spec, idx, axis=-1)           # [..., k] complex

    # one-sided energy correction
    F = n // 2 + 1
    is_dc = (idx == 0)
    is_nyq = (idx == (F - 1)) & ((n % 2) == 0)
    w = jnp.where(is_dc | is_nyq, 1.0, jnp.sqrt(2.0)).astype(x.dtype)
    picked = picked * w

    y = jnp.concatenate([picked.real, picked.imag], axis=-1)  # [..., 2k]
    return jnp.sqrt(n / y.shape[-1]).astype(x.dtype) * y



def dimred_hadamard(rng, grads, k):
  rng = jax.random.key(0)
  rng, key = jax.random.split(rng)
  n = grads.shape[-1]
  signs, idx, n2 = fjlt_hadamard_init(rng, n, k)
  return fjlt_hadamard_apply(grads, signs, idx, n2)



def fft_sketch(grads, signs, idx):
  """
  grads: [..., n] real
  signs: [n] ±1
  idx: [k] indices into rfft bins (0..n//2)
  returns: [..., k] real sketch
  """
  n = grads.shape[-1]
  x = grads * signs  # broadcast over leading dims

  # rFFT -> take selected bins
  spec = jnp.fft.rfft(x, axis=-1, norm="ortho")             # [..., F] complex
  picked = jnp.take(spec, idx, axis=-1)       # [..., k] complex

  # map complex -> real k dims (simple choice: real part)
  # scale to roughly preserve dot products
  return jnp.sqrt(n / idx.shape[0]) * picked.real

def fft_sketch_2k(grads, signs, idx):
    n = grads.shape[-1]
    x = grads * signs
    spec = jnp.fft.rfft(x, axis=-1, norm="ortho")
    picked = jnp.take(spec, idx, axis=-1)  # [..., k] complex
    feats = jnp.concatenate([picked.real, picked.imag], axis=-1)  # [..., 2k]
    return jnp.sqrt(n / idx.shape[0]) * feats


def fjlt_hadamard_init(rng, n, k, dtype=jnp.float32):
    """
    Create a fixed SRHT/FJLT sketch:
      y = sqrt(n2/k) * P * (H/sqrt(n2)) * D * pad(g)

    Returns:
      signs: [n2] in `dtype` with entries ±1
      idx:   [k] int32 indices in [0, n2)
      n2:    padded length (power of 2)
    """
    n2 = 1 << (n - 1).bit_length()  # next power of 2 >= n
    rng, k1, k2 = jax.random.split(rng, 3)

    signs = jax.random.rademacher(k1, (n2,), dtype=jnp.int8).astype(dtype)
    idx = jax.random.choice(k2, n2, shape=(k,), replace=False).astype(jnp.int32)
    return signs, idx, n2


def fjlt_hadamard_apply(grads, signs, idx, n2):
    """
    grads: [..., n]
    signs: [n2]
    idx:   [k]
    n2:    power-of-2 padding length
    returns: [..., k]
    """
    n = grads.shape[-1]
    if n > n2:
        raise ValueError(f"n ({n}) must be <= n2 ({n2}).")

    # Pad to n2
    pad_width = [(0, 0)] * (grads.ndim - 1) + [(0, n2 - n)]
    x = jnp.pad(grads, pad_width)

    # Multiply by Rademacher signs (D)
    x = x * signs.astype(x.dtype)

    # Fast Walsh–Hadamard Transform along last axis (unnormalized)
    # Unroll stages in Python so reshape sizes are static.
    logn = int(math.log2(n2))
    x2 = x.reshape((-1, n2))  # [B, n2]
    B = x2.shape[0]

    for s in range(logn):
        h = 1 << s  # python int
        x2 = x2.reshape((B, n2 // (2 * h), 2, h))
        u = x2[:, :, 0, :]
        v = x2[:, :, 1, :]
        x2 = jnp.concatenate([u + v, u - v], axis=2).reshape((B, n2))

    x2 = x2.reshape(x.shape)

    # Orthonormalize: H / sqrt(n2)
    x2 = x2 * (1.0 / jnp.sqrt(jnp.array(n2, dtype=x2.dtype)))

    # Subsample (P) and scale sqrt(n2/k)
    y = jnp.take(x2, idx, axis=-1)
    return y * jnp.sqrt(jnp.array(n2 / idx.shape[0], dtype=y.dtype))


def dimred_global_packed(grads, k: int):
  # grads leaves: [(B, D1), (B, D2), ...] -> concat: (B, Dtot)
  X = pack_pytree(grads)          # (B, Dtot)
  scores = X.mean(0)                            # (Dtot,)
  k = int(min(k, X.shape[1]))                   # static for jit
  idx = jax.lax.top_k(scores, k)[1]             # (k,)
  return jnp.take(X, idx, axis=1)               # (B, k)

def dimred_topk(grads, k: int):
  # grads leaves: [(B, D1), (B, D2), ...] -> concat: (B, Dtot)
  X = pack_pytree(grads)          # (B, Dtot)
  return jax.lax.top_k(X, k)[0]
