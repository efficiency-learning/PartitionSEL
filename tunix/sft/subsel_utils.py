from __future__ import annotations

from typing import Any, Tuple, NamedTuple, Callable

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from functools import partial
import optax
from tunix.sft import utils
import chex
from tunix.sft.subsel.subsel import *
from tunix.sft.subsel.utils import *
from tunix.sft.subsel.dimred import *
from tunix.sft.subsel.grads import *
from tunix.sft.subsel.grads import grad_filter

def _dbg(verbose, fmt, *args):
  """Conditional jax.debug.print — no-op when verbose is False."""
  if verbose:
    jax.debug.print(fmt, *args)

# Module-level verbose flag — set from task_cfg["subsel"]["verbose"] in subset_select.
# Defaults to True (preserve existing behavior). Set to False for timing experiments.
_VERBOSE = True


@partial(
  nnx.jit, 
  static_argnames=("task_cfg"), 
  donate_argnames=("model", "project", "optimizer_head", "cache_train", "cache_val")
)
def process_gradsv2(model, project, optimizer_head, cache_train, 
                    cache_val, inputs_train, inputs_val, moments_train, moments_val,
                    step, task_cfg, rng):
  loss = {}
  grads_train, grads_val = None, None
  _verbose = task_cfg.get("verbose", True)
  ret_meta = {
     "moments_train": None,
     "moments_val": None,
  }
  lowpass = task_cfg["grads"]["lowpass_adam"]
  chunk_size = task_cfg["grads"]["chunk_size"]
  grad_layer = task_cfg["grads"]["grad_layer"]

  grads_train, meta_train = per_ex_grads(model, step, inputs_train, task_cfg, grad_layer,
                                         rng, lowpass=lowpass,  moments=moments_train, chunk_size=chunk_size)
  if task_cfg["subsel"]["val_anchors"]:
    grads_val, meta_val = per_ex_grads(model, step, inputs_val, task_cfg, grad_layer, rng,
                                       lowpass=lowpass, moments=moments_val, chunk_size=chunk_size)
  else:
    grads_val, meta_val = grads_train, meta_train

  ret_meta["moments_train"] = meta_train["moments"]
  ret_meta["moments_val"] = meta_val["moments"]

  grads_train = post_process_grad(grads_train, task_cfg)
  grads_val = post_process_grad(grads_val, task_cfg)

  S_tt = gram_linear(grads_train)
  S_tv = gram_linear(grads_train, grads_val)

  return S_tt, S_tv, grads_train, grads_val, loss, ret_meta


def updatedict(tag, kv, mega):
   for k in kv.keys():
      mega[f"{k}_{tag}"] = kv[k]
   return mega


def domainwise_select(S_tt, S_tv, ratio, inputs_meta, mode, num_train_sources, grads, lr=1.0, _verbose=True):
  D_tt = 1- S_tt
  k_eff = inputs_meta["budget"]
  sources = inputs_meta["sources"]
  out_offsets = inputs_meta["out_offsets"]
  flag = False
  triggers = ["facloc", "gradnorm"]
  if mode in triggers: flag = True
  idx, _ = select_per_class(S_tt, S_tv, D_tt, ratio, sources, 
                   k_eff, out_offsets, optim_name=mode,
                   max_classes= num_train_sources, apply_source_mask_on_target=flag,
                   verbose=_verbose, lr=lr)
  _dbg(_verbose, "conflicting-pairs={}", conflicting(grads, idx, ratio))
  _dbg(_verbose, "domainwise {} {} \n tgt_mask={}", mode, idx, flag)
  return idx



@partial(
  nnx.jit, 
  static_argnames=("ratio", "mode", "config", "num_train_sources"),
  donate_argnames=("model", "project", "optimizer_head", "cache_train", "cache_val")
)
def subset_select(model, num_train_sources, project, optimizer_head, cache_train, 
                  cache_val, config, moments, step, ratio, mode, inputs, val_batch, lr, rng, cls_meta):
# def subset_select(config, model, project, project2, optimizer, optimizer_last, optimizer_head, optimizer_head2, 
#                   moments, step, ratio, mode, inputs, val_batch, lr, rng, cache_train, cache_val):
  # jax.debug.print("input kets befor{}e", inputs.keys())
  bs = jax.tree.leaves(inputs)[0].shape[0]
  task_cfg = config["task_config"]["config"]
  _verbose = task_cfg["subsel"].get("verbose", True)
  subsel_enabled = config["subset_select"]["enabled"]

  inputs_meta = None
  if "meta" in inputs.keys():
    meta = inputs.pop("meta")
    # budget = jnp.where(budget == 1, budget, budget // 2)
    inputs_meta = dict(
          budget=meta["k_eff"][0], # [MC]
          sources=meta["sources"],
          out_offsets=meta["out_offsets"][0],
          # class_valid_mask=jnp.transpose(meta["class_valid_mask"]), # [MC, N]
          # out_offsets=meta["out_offsets"][0],  # [MC]
      )
    # jax.debug.print("input kets after {} =={}", inputs_meta["budget"].shape, meta["k_eff"].shape)
  if config["subset_select"]["enabled"] and task_cfg["subsel"]["val_anchors"]:
    if "meta" in val_batch.keys():
      meta = val_batch.pop("meta")
      val_meta = dict(sources=meta["sources"],)
    else:
      val_meta = dict(sources=jnp.ones(bs, dtype=jnp.int32),)
  else:
     val_meta = inputs_meta
  
  mt_train_batch, mt_val_batch = None, None
  aux = {}

  if mode not in ["full", "random"]:
    S_tt, S_tv, grads, grads_val, aux, aux_mom = process_gradsv2(model, project, optimizer_head, cache_train, 
                    cache_val, inputs, val_batch, moments["train"], moments["val"], step, task_cfg, rng)
    mt_train_batch, mt_val_batch = aux_mom["moments_train"], aux_mom["moments_val"]
    anchors = grads_val if task_cfg["subsel"]["val_anchors"] else grads
  
  if mode not in ["full", "random", "joint", "iwd"] and task_cfg["subsel"]["domainwise"]:
    idx = domainwise_select(S_tt, S_tv, ratio, inputs_meta, mode, num_train_sources, grads, lr=lr, _verbose=_verbose)
  
  elif mode == "full" or not subsel_enabled:
    idx = jnp.arange(bs)
    ratio = bs

  elif mode == "random":
    rng, key = jax.random.split(rng)
    idx = jax.random.permutation(key, jnp.arange(bs))[:ratio]

  elif mode == "gradnorm":
    norms = jnp.diag(S_tt).astype(jnp.float32)
    _, idx = jax.lax.top_k(norms, ratio)
    _dbg(_verbose, "gradnorm {} {}", norms, jnp.sort(idx))

  elif mode == "uniprot":
    # jax.debug.print("model {}", jax.tree.map(lambda x: x.shape, model))
    sims = S_tv.astype(jnp.float32)
    dist = jnp.max(sims) - sims
    idx = greedy_fairot(sims, dist, ratio, reg=task_cfg["subsel"]["reg"], iters=task_cfg["subsel"]["iters"], limit=ratio)
    _dbg(_verbose, "uniprot idx {} {}", sims.shape, idx)

  elif(mode == "facloc"):
    sims = S_tv.astype(jnp.float32)
    idx = facility_location_old2(sims, ratio)

  elif(mode == "greats"):
    # jax.debug.print("lr {}", lr)
    
    sims = S_tv.astype(jnp.float32)
    scores = jnp.mean(sims, axis=1)
    interaction_matrix = S_tt.astype(jnp.float32)
    lr1 = lr
    lr2 = lr**2
    idx = greats_selection(lr1*scores, lr2*interaction_matrix, ratio, limit=None)
    _dbg(_verbose, "greats idx {}", idx)

  elif(mode == "iwd"):
    sources = inputs_meta["sources"]
    src_v = val_meta["sources"]

    anchors = wt_mean(task_cfg, num_train_sources, src_v, anchors)
    budget = inputs_meta["budget"]

    sims = gram_linear(grads, anchors).astype(jnp.float32)
    scores = jnp.mean(sims, axis=1)
    full_interaction_matrix = S_tt.astype(jnp.float32)

    # ================================================================
    # IWD = Independent Weighted Domain-wise selection.
    #
    # This is the weighted analogue of ID.  It should learn continuous
    # prototype weights independently inside each source/domain, while
    # still honoring the same per-domain budgets.
    #
    # Implementation trick: we reuse joint_subsel, but intentionally make
    # K block diagonal by zeroing ONLY cross-domain interactions:
    #
    #   K_iwd[i, j] = <g_i, g_j> if source_i == source_j
    #                 0          otherwise
    #
    # Therefore the utility decomposes as sum_c U_c(w_c), exactly the IWD
    # baseline.  Unlike the old joint bug, KEEP each block diagonal term
    # K_ii = ||g_i||^2 so APGD has self-curvature and finite weights.
    # ================================================================
    same_domain = sources[:, None] == sources[None, :]
    interaction_matrix = jnp.where(same_domain, full_interaction_matrix, 0.0)

    lr = jax.lax.cond(lr > 1e-12, lambda _: lr, lambda _: 1e-12, None)
    lr1 = lr
    lr2 = lr**2
    weighted_scores = lr1 * scores + 0.5 * lr2 * jnp.diag(interaction_matrix)
    weighted_interaction_matrix = lr2 * interaction_matrix
    apgd_lr = 1/(jax.numpy.linalg.matrix_norm(weighted_interaction_matrix) + 1e-8)
    mask, utilities, weights = joint_subsel(rng, sources, budget, weighted_scores,
                    apgd_lr, weighted_interaction_matrix, cls_meta["prev_utils"], task_cfg["subsel"]["apdg_iters"])
    idx = jnp.nonzero(mask, size=ratio)[0]

    aux["iwd_weight_sum"] = jnp.sum(weights)
    aux["iwd_weight_max"] = jnp.max(weights)
    _dbg(_verbose, "IWD Learning rate {}", apgd_lr)
    _dbg(_verbose, "conflicting-pairs={}", conflicting(grads, idx, ratio))
    _dbg(_verbose, "iwd idx \n bin={} \n bud={} {} \n idx={} #############################",
                        jnp.bincount(sources, length=num_train_sources), budget, jnp.sum(budget), idx)

  elif(mode == "joint"):
    # Temporarily disabled: keep original joint logic below untouched.
    # raise NotImplementedError("mode='joint' is temporarily disabled")

    # start = time.perf_counter()
    sources = inputs_meta["sources"]
    src_v = val_meta["sources"]

    anchors = wt_mean(task_cfg, num_train_sources, src_v, anchors)

    budget = inputs_meta["budget"]
    # make new budget where classes with only 1 el are always selected
    # budget = jnp.where(hist == 1, jnp.zeros_like(budget), budget) # [MC]

    sims = gram_linear(grads, anchors).astype(jnp.float32)
    scores = jnp.mean(sims, axis=1)
    interaction_matrix = S_tt.astype(jnp.float32)
    # interaction_matrix = interaction_matrix*(1 - jnp.eye(sims.shape[0], sims.shape[0]))
    lr = jax.lax.cond(lr > 1e-12, lambda _: lr, lambda _: 1e-12, None)
    lr1 = lr
    lr2 = lr**2
    # ================================================================
    # CRITICAL WEIGHTED-OBJECTIVE DETAIL:
    #
    # The weighted PartitionSel/APGD objective is
    #
    #   U(w) = <w, lr * score + 0.5 * lr^2 * diag(K)>
    #          - 0.5 * w^T (lr^2 * K) w
    #
    # where K_ij = <g_i, g_j>. KEEP THE DIAGONAL IN K.
    #
    # For binary GREATS-style greedy it is tempting to zero the diagonal
    # because selected items are manually removed. That is wrong here:
    # joint_subsel refits continuous nonnegative weights with APGD, and
    # diag(K) = ||g_i||^2 is the self-curvature term that prevents a
    # selected item with positive score from receiving an unbounded weight.
    #
    # In short: do not replace K by K - diag(K) for weighted joint.
    # ================================================================
    weighted_scores = lr1 * scores + 0.5 * lr2 * jnp.diag(interaction_matrix)
    weighted_interaction_matrix = lr2 * interaction_matrix
    # apgd_lr = 1e-4
    beta = task_cfg["curricullum"]["beta"]
    lamb = task_cfg["curricullum"]["lamb"]
    max_iters = task_cfg["subsel"]["apdg_iters"]
    budget = curr_budget(task_cfg, lamb, sources, budget, cls_meta["gain"], ratio, rng)
    apgd_lr = 1/(jax.numpy.linalg.matrix_norm(weighted_interaction_matrix) + 1e-8)
    mask, utilities, weights = joint_subsel(rng, sources, budget, weighted_scores,
                    apgd_lr, weighted_interaction_matrix ,cls_meta["prev_utils"], max_iters)

    final_utils = per_class_utilities(grads, anchors, src_v, num_train_sources, lr1, lr2, interaction_matrix, weights)
    # 1. Compute temp (same as before)
    temp = jnp.where(
        cls_meta["prev_utils"] == 0,
        0.0,
        (final_utils - cls_meta["prev_utils"]) / jnp.abs(cls_meta["prev_utils"])
    )

    # 2. Update gain only at src_v indices
    gain = cls_meta["gain"]

    # src_v must be an int array of class indices in [0, num_train_sources)
    # gain = gain.at[src_v].set(
    #     beta * temp[src_v] + (1.0 - beta) * gain[src_v]
    # )
    gain = beta * temp + (1.0 - beta) * gain

    cls_meta["gain"] = gain
    cls_meta["prev_utils"] = final_utils
    # cls_meta["prev_utils"] = utilities
    import sys
    jnp.set_printoptions(threshold=sys.maxsize)
    idx = jnp.nonzero(mask, size=ratio)[0]

    # jax.lax.cond(
    #   step % 8 == 0,
    #   lambda _: jax.debug.print("utilities #START\n{}\n#END", utilities[:ratio]),
    #   lambda _: None,
    #   None
    # )
    # jax.debug.print("weights {} {} {}",  weights, utilities.shape, ratio)

    _dbg(_verbose, "Learning rate {}", apgd_lr)
    _dbg(_verbose, "conflicting-pairs={}", conflicting(grads, idx, ratio))
    _dbg(_verbose, "class_utils={}", final_utils)
    _dbg(_verbose, "gain={}", cls_meta["gain"])
    _dbg(_verbose, "src_v={}", src_v)
    _dbg(_verbose, "prev_utils[0]: {}", cls_meta["prev_utils"][0])
    _dbg(_verbose, "final_utils[0]: {}", final_utils[0])
    _dbg(_verbose, "temp[0]: {}", temp[0])
    _dbg(_verbose, "gain[0] before: {}", cls_meta["gain"][0])
    _dbg(_verbose, "lowpass={}", task_cfg["grads"]["lowpass_adam"])
    # add els to mask where class size == 1
    # mask = jnp.where()
    cnt = jnp.bincount(sources, length=num_train_sources)
    _dbg(_verbose, "joint idx \n bin={} \n bud={} {} \n idx={} #############################",
                        cnt, budget, jnp.sum(budget), idx)

  
  rng, key = jax.random.split(rng)
  idx = jax.random.permutation(key, idx)
  subset = jax.tree.map(lambda x: x[idx], inputs)
  # subset = jax.tree.map(lambda x : x[idx], inputs)
  # loss_mask = jnp.zeros_like(norms).at[idx].set(True)
  # loss_mask = jnp.ones((ratio))
  loss_mask = jnp.zeros((bs)).at[idx].set(1)
  # subset = jax.tree.map(lambda x: x.block_until_ready(), subset)
  # state = nnx.state((model, project, optimizer_head))
  def choose_mt(mt_batch):
    mu, nu = None, None
    if mt_batch is not None:
      mu_batch, nu_batch = mt_batch
      mu = jax.tree.map(lambda x: x[idx].mean(0), mu_batch)
      nu = jax.tree.map(lambda x: x[idx].mean(0), nu_batch)
    return (mu, nu)
  
  # lp_val = task_cfg["grads"]["lowpass_val"]
  if task_cfg["grads"]["lowpass_adam"]:
    val_smooth = task_cfg["grads"]["val_smooth"]
    if val_smooth == "val": pass
    elif val_smooth == "train": mt_val_batch = mt_train_batch

  moments = {
    "train": choose_mt(mt_train_batch),
    "val": choose_mt(mt_val_batch),
  }
  # ret1, ret2 = _train_step(model, optimizer, inputs, _steps, loss_mask)
  return subset, moments, loss_mask, aux, cls_meta


@partial(jax.jit, static_argnames=("optim_name", "out_len"))
def get_optim(optim_name, S_tt, S_tv, D_tt, out_len, limit, src_mask=None, tgt_mask=None, lr=1.0):
    """
    Unified contract (enforced here, not per method):
      - Returns exactly length-M int32 vector of GLOBAL indices.
      - Positions >= limit (if given) are forced to -1.
      - Indices must be in [0, N); out-of-bounds are set to -1.
      - If src_mask is provided, any index with src_mask[idx] == False is set to -1.
      - When there are no candidates or limit <= 0, returns all -1.
      - Padding uses -1.

    Notes:
      - This wrapper passes masks and limit through to the underlying method,
        but *also* sanitizes its outputs to the contract above.
      - Assumes underlying methods already return a length-M vector (padded).
        If they don't, make them do so (static shape needed for JIT).
    """
    N = S_tt.shape[0]
    cols = jnp.arange(out_len, dtype=jnp.int32)

    # Default masks: True everywhere if None
    if src_mask is None:
        src_mask = jnp.ones((N,), dtype=bool)
    if tgt_mask is None:
        tgt_mask = jnp.ones((N,), dtype=bool)

    # Candidates exist?
    has_cand = jnp.logical_and(src_mask, tgt_mask).any()

    # Normalize limit: if None -> M; clamp to [0, out_len]
    # limit = jnp.where(limit is None, out_len, limit)
    limit = jnp.clip(jnp.asarray(limit, dtype=jnp.int32), 0, out_len)

    def _call_inner():
        if optim_name == "uniprot":
            out = greedy_fairot(
                S_tt, D_tt, out_len, reg=1e-2, iters=20,
                src_mask=src_mask, tgt_mask=tgt_mask, limit=limit
            )
        elif optim_name == "greats":
            lr_safe = jnp.where(lr > 1e-12, lr, 1e-12)
            out = greats_selection(
                lr_safe * S_tv.mean(1), (lr_safe ** 2) * S_tt, out_len,
                source_mask=src_mask, limit=limit
            )
        elif optim_name == "facloc":
            out = facility_location(
                S_tt, out_len, src_mask=src_mask, tgt_mask=tgt_mask, limit=limit
            )
        elif optim_name == "gradnorm":
          norms = jnp.diag(S_tt).astype(jnp.float32)
          scores = jnp.where(src_mask, norms, -jnp.inf)
          _, out = jax.lax.top_k(scores, out_len)
        else:
          raise ValueError(f"Unknown optimizer: {optim_name}")
        return out

    def _all_neg1():
        return jnp.full((out_len,), -1, dtype=jnp.int32)

    run_flag = jnp.logical_and(has_cand, limit > 0)
    sel = jax.lax.cond(run_flag, _call_inner, _all_neg1)      # [M], dtype may vary

    # ---- Contract enforcement (centralized) ----
    sel = jnp.asarray(sel, dtype=jnp.int32)                   # dtype normalize

    # Enforce limit positionally: slots >= limit are padding
    sel = jnp.where(cols < limit, sel, -1)

    # In-bounds check
    inb = (sel >= 0) & (sel < N)

    # Respect src_mask: only keep selections with src_mask=True
    # Guard gather with in-bounds mask to avoid indexing with -1
    src_ok = jnp.where(inb, src_mask[sel], False)

    # Final sanitize: invalid or masked-out -> -1
    sel = jnp.where(inb & src_ok, sel, -1)

    return sel


from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=("out_len", "optim_name", "max_classes"))
def select_per_class213(
    S_tt, S_tv, D_tt,
    out_len: int,
    sources,               # [N]
    k_eff,                 # [MC], sum == out_len
    out_offsets,           # [MC], prefix sum of k_eff
    optim_name: str,
    max_classes: int,
):
    """
    Fresh version:
      1) run optimizer per class -> sel[c, :] (fixed length out_len, padded with -1)
      2) pack sel into a single output using prefix-sum slices:
            out[out_offsets[c] : out_offsets[c] + k_eff[c]] = sel[c, :k_eff[c]]
    """

    cols = jnp.arange(out_len, dtype=jnp.int32)

    class_ids = jnp.arange(max_classes, dtype=sources.dtype)          # [MC]
    class_valid_mask = (sources[None, :] == class_ids[:, None])       # [MC, N]

    has_cand  = jnp.any(class_valid_mask, axis=1)                     # [MC]
    run_flags = has_cand & (k_eff > 0)                                # [MC]

    def per_class(k_i, mask_i, run_i):
        def _do():
            return get_optim(
                optim_name, S_tt, S_tv, D_tt,
                out_len,
                limit=k_i,
                src_mask=mask_i,
                tgt_mask=None,
                lr=lr,
            )  # [out_len], padded with -1
        def _zero():
            return jnp.full((out_len,), -1, dtype=jnp.int32)
        return jax.lax.cond(run_i, _do, _zero)

    sel = jax.vmap(per_class, in_axes=(0, 0, 0))(k_eff, class_valid_mask, run_flags)  # [MC, out_len]
    _dbg(verbose, "sel {}", sel)
    # Output buffer
    out = jnp.full((out_len,), -1, dtype=jnp.int32)

    # Pack by disjoint slices. This is the "prefix sum packing" you mean.
    def body(c, out_buf):
        k = k_eff[c]
        start = out_offsets[c]
        # take first k entries from sel[c]
        vals = sel[c, :k]  # shape [k] (dynamic k)
        # write into out[start : start+k]
        out_buf = out_buf.at[start:start + k].set(vals)
        jax.lax.dynamic_update_slice(out_buf, )
        return out_buf

    out = jax.lax.fori_loop(0, max_classes, body, out)
    return out, None


from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=("out_len", "optim_name", "max_classes", "apply_source_mask_on_target", "verbose"))
def select_per_class(
    S_tt, S_tv, D_tt,
    out_len: int,          # global selection budget
    sources,               # [N] int32 class id in [0..MC-1]
    k_eff,                 # [MC] int32, sum(k_eff) == out_len (or <= out_len if you allow slack)
    out_offsets,           # [MC] int32, prefix sum of k_eff (defines disjoint slices)
    optim_name: str,       # static
    max_classes: int,      # MC (static)
    apply_source_mask_on_target: bool,
    verbose: bool = True,
    lr: float = 1.0,
):
    """
    Returns:
      orders_out: [out_len] int32

    High-level logic:
      1) For each class c, run get_optim restricted to that class => sel[c, :]
         where sel[c, :] is fixed-length [out_len] padded with -1.
      2) Pack into a single [out_len] array using prefix-sum ownership:
           class c owns positions [out_offsets[c], out_offsets[c] + k_eff[c])
         For each output position p:
           find owning class c
           j = p - out_offsets[c]
           output[p] = sel[c, j]
      This avoids dynamic slice writes and avoids scatter collisions entirely.
    """

    # ---------- 1) Per-class optimizer runs (fixed shapes) ----------

    class_ids = jnp.arange(max_classes, dtype=sources.dtype)          # [MC]
    class_valid_mask = (sources[None, :] == class_ids[:, None])       # [MC, N]

    has_cand  = jnp.any(class_valid_mask, axis=1)                     # [MC]
    run_flags = has_cand & (k_eff > 0)                                # [MC]

    def per_class(k_i, mask_i, run_i):
        # If runnable: run optimizer on this class only.
        def _do():
            return get_optim(
                optim_name, S_tt, S_tv, D_tt,
                out_len,
                src_mask=mask_i,
                tgt_mask=mask_i if apply_source_mask_on_target else None,
                limit=k_i,
                lr=lr,
            )  # [out_len], padded with -1
        # If not runnable: all -1.
        def _zero():
            return jnp.full((out_len,), -1, dtype=jnp.int32)
        return jax.lax.cond(run_i, _do, _zero)

    sel = jax.vmap(per_class, in_axes=(0, 0, 0))(k_eff, class_valid_mask, run_flags)  # [MC, out_len]
    # jax.debug.print("sel {}", sel)


    # ---------- 2) Prefix-sum packing as a gather (JIT-safe) ----------

    # Each class owns a half-open interval:
    #   [start[c], end[c]) where start = out_offsets, end = out_offsets + k_eff.
    start = out_offsets.astype(jnp.int32)                # [MC]
    end   = (out_offsets + k_eff).astype(jnp.int32)      # [MC]

    # For each output position p, determine which class owns it.
    p = jnp.arange(out_len, dtype=jnp.int32)             # [out_len]

    # owns[c, p] = True iff p is in class c's slice.
    # If out_offsets/k_eff are correct prefix sums, each column p has exactly one True.
    owns = (p[None, :] >= start[:, None]) & (p[None, :] < end[:, None])  # [MC, out_len]

    # If sum(k_eff) == out_len, every p should be owned. If you allow slack, some may be unowned.
    has_owner = jnp.any(owns, axis=0)                    # [out_len]

    # Because slices are disjoint, argmax picks the unique owning class when has_owner is True.
    owner = jnp.argmax(owns, axis=0).astype(jnp.int32)   # [out_len] in [0..MC-1] (arbitrary if unowned)

    # Local index within that class slice: j = p - start[owner].
    j = (p - start[owner]).astype(jnp.int32)             # [out_len]

    # Gather the selected element: sel[owner[p], j[p]].
    gathered = sel[owner, j]                             # [out_len]

    # If a position is unowned, force -1. Also keep only valid indices (>=0).
    orders_out = jnp.where(has_owner, gathered, -1).astype(jnp.int32)

    return orders_out, None



