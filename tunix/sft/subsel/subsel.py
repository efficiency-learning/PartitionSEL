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
import chex
import time
import numpy as np


def _partition_accounting(selected_mask, part_id, budget):
    """
    used[p] = how many elements already selected in partition p
    rem[p]  = remaining capacity in partition p (budget - used)
    cnt[p]  = how many total elements exist in partition p
    """
    P = budget.shape[0]

    # bincount over part_id; selected_mask acts as 0/1 weights
    used = jnp.bincount(
        part_id,
        weights=selected_mask.astype(jnp.int32),
        length=P,
    ).astype(budget.dtype)

    rem = budget - used

    cnt = jnp.bincount(part_id, length=P).astype(budget.dtype)
    return used, rem, cnt



def _sample_uniform_from_mask(key, pick_mask):
    """
    Uniformly sample an index from the True positions of pick_mask.

    Implemented as categorical over logits:
      logits[i] = 0     if allowed
               = -inf  otherwise
    This makes the distribution uniform over allowed indices.

    If pick_mask is empty:
      did_pick = False and we return dummy index 0 (caller must gate on did_pick).
    """
    did_pick = jnp.any(pick_mask)
    logits = jnp.where(pick_mask, 0.0, -jnp.inf)

    key, subkey = jax.random.split(key)
    picked = jax.lax.cond(
        did_pick,
        lambda k: jax.random.categorical(k, logits, axis=0).astype(jnp.int32),
        lambda k: jnp.int32(0),
        subkey,
    )
    return key, did_pick, picked


def test_shortlist(part_id, feasible, rem, part_s, feas_s, cap_s, short_s, sort_idx):
  P = rem.shape[0]
  # part_id range and feasible implies cap>0
  bad_pid = (part_id < 0) | (part_id >= P)
  jax.lax.cond(
      jnp.any(bad_pid),
      lambda _: jax.debug.print(
          "SHORTLIST CHECK FAIL: part_id out of range. P={} part_id(min,max)=({}, {}) bad_mask={}",
          P, jnp.min(part_id), jnp.max(part_id), bad_pid.astype(jnp.int32)
      ),
      lambda _: None,
      operand=None,
  )

  bad_feas = feas_s & (cap_s <= 0)
  jax.lax.cond(
      jnp.any(bad_feas),
      lambda _: jax.debug.print(
          "SHORTLIST CHECK FAIL: feasible item with cap<=0 in sorted space. cap_s={} bad_feas_mask={}",
          cap_s, bad_feas.astype(jnp.int32)
      ),
      lambda _: None,
      operand=None,
  )

  # Per-partition counts in original space
  feas_per_part = jnp.bincount(part_id, weights=feasible.astype(jnp.int32), length=P).astype(jnp.int32)
  expected_per_part = jnp.minimum(rem.astype(jnp.int32), feas_per_part)
  expected = jnp.sum(expected_per_part)

  # Actual shortlist-per-part from scan result (sorted space)
  short_per_part = jnp.bincount(part_s, weights=short_s.astype(jnp.int32), length=P).astype(jnp.int32)
  actual = jnp.sum(short_s.astype(jnp.int32))

  # Main invariant + detailed dump when it fails
  jax.lax.cond(
      expected != actual,
      lambda _: jax.debug.print(
          "SHORTLIST BUG (sorted scan): expected={} actual={} | rem={} feas_per_part={} expected_per_part={} short_per_part={} | part_s={} feas_s={} cap_s={} short_s={} sort_idx={}",
          expected, actual,
          rem, feas_per_part, expected_per_part, short_per_part,
          part_s, feas_s.astype(jnp.int32), cap_s, short_s.astype(jnp.int32),
          sort_idx
      ),
      lambda _: None,
      operand=None,
  )

  # Helpful always-on summary (comment out if too noisy)
  # jax.debug.print(
  #     "SHORTLIST DBG: #feasible={} #shortlisted={} has_pos={}",
  #     jnp.sum(feasible.astype(jnp.int32)),
  #     actual,
  #     jnp.any(feasible & (proxy > 0.0)),
  # )

@partial(jax.jit, static_argnames=("scores_relu"))
def _shortlist_sorted(part_id, rem, proxy, feasible, scores_relu):
    """
    shortlist builder in *sorted space*.

    What it does:
      1) score[i] = ReLU(proxy[i])
      2) sort items by (partition asc, score desc) using two stable argsorts
      3) scan left-to-right; for each partition p keep the first rem[p] feasible items

    Returns:
      sort_idx : [N]  (sorted position -> original index)
      part_s   : [N]  part_id in sorted order
      feas_s   : [N]  feasible in sorted order
      short_s  : [N]  shortlist mask in sorted order
      has_pos  : bool any feasible proxy>0 in original space
      scores   : [N]  ReLU(proxy) in original order (for logging/inspection)
    """
    N = part_id.shape[0]
    P = rem.shape[0]

    scores = jnp.where(scores_relu, jnp.maximum(proxy, 0.0), proxy)
    # scores = jnp.maximum(proxy, 0.0)

    NEG = jnp.array(-1e30, dtype=scores.dtype)
    key_score = jnp.where(feasible, scores, NEG)

    # Two-pass stable sort: score desc, then partition asc (stable)
    idx_score = jnp.argsort(-key_score, stable=True)
    idx_part  = jnp.argsort(part_id[idx_score], stable=True)
    sort_idx  = idx_score[idx_part]

    part_s = part_id[sort_idx]
    feas_s = feasible[sort_idx].astype(jnp.bool_)
    cap_s  = rem[part_s].astype(jnp.int32)

    def scan_body(counts, x):
        p, feas, cap = x
        c = counts[p]
        take = feas & (c < cap)
        counts = counts.at[p].add(take.astype(jnp.int32))
        return counts, take

    counts0 = jnp.zeros((P,), dtype=jnp.int32)
    counts_f, short_s = jax.lax.scan(
        scan_body,
        counts0,
        (part_s.astype(jnp.int32), feas_s, cap_s),
        length=N,
    )


    has_pos = jnp.any(feasible & (proxy > 0.0))
    test_shortlist(part_id, feasible, rem, part_s, feas_s, cap_s, short_s, sort_idx)

    dbg = partial(_dbg, False)
    dbg("scores {}", scores)
    dbg("has_pos {}", has_pos)
    dbg("sort_idx {}", sort_idx)
    dbg("part_s {}", part_s)
    dbg("feas_s {}", feas_s.astype(jnp.int32))
    dbg("short_s {}", short_s.astype(jnp.int32))
    dbg("cap_s {}", cap_s)

    # Fallback logic
    use_fallback = (~has_pos) & jnp.any(feasible)
    use_fallback = jnp.where(scores_relu, use_fallback, False)

    dbg(
        "fallback decision: has_pos={} any_feasible={} -> use_fallback={}",
        has_pos,
        jnp.any(feasible),
        use_fallback,
    )

    pick_s = jnp.where(use_fallback, feas_s, short_s)

    dbg("pick_s (sorted mask) {}", pick_s.astype(jnp.int32))
    dbg("pick_s count {}", jnp.sum(pick_s.astype(jnp.int32)))


    return sort_idx, pick_s, use_fallback

def _sample_uniform_from_mask_sorted(key, pick_mask_s):
    did_pick = jnp.any(pick_mask_s)
    logits = jnp.where(pick_mask_s, 0.0, -jnp.inf)
    key, subkey = jax.random.split(key)
    picked_s = jax.lax.cond(
        did_pick,
        lambda k: jax.random.categorical(k, logits, axis=0).astype(jnp.int32),
        lambda k: jnp.int32(0),
        subkey,
    )
    return key, did_pick, picked_s


def _dbg(enabled, fmt, *args):
    return jax.lax.cond(
        enabled,
        lambda _: jax.debug.print(fmt, *args),
        lambda _: None,
        operand=None,
    )


@jax.jit
def select_one_partition_matroid_toprelu(key, selected_mask, part_id, budget, proxy):
    dbg = partial(_dbg, False)

    # Partition accounting
    used, rem, cnt = _partition_accounting(selected_mask, part_id, budget)

    dbg("SELECT STEP -----------------------------")
    dbg("part_id {}", part_id)
    dbg("selected_mask {}", selected_mask.astype(jnp.int32))
    dbg("used {}", used)
    dbg("budget {}", budget)
    dbg("rem {}", rem)
    dbg("cnt {}", cnt)

    # Feasibility
    rem_i = rem[part_id]
    feasible = (~selected_mask) & (rem_i > 0)

    dbg("rem_i {}", rem_i)
    dbg("feasible {}", feasible.astype(jnp.int32))
    dbg(
        "feasible stats: #feasible={} #not_sel={} rem_i(min,max)=({}, {})",
        jnp.sum(feasible.astype(jnp.int32)),
        jnp.sum((~selected_mask).astype(jnp.int32)),
        jnp.min(rem_i),
        jnp.max(rem_i),
    )

    # Shortlist (sorted space)
    sort_idx, pick_s, use_fallback = _shortlist_sorted(
        part_id, rem, proxy, feasible, scores_relu=False
    )

    # Sampling (sorted space)
    key, did_pick, picked_s = _sample_uniform_from_mask_sorted(key, pick_s)
    picked = sort_idx[picked_s]

    dbg(
        "sampling: did_pick={} picked_s={} picked(orig)={}",
        did_pick,
        picked_s,
        picked,
    )

    dbg(
        "picked feasibility check: feasible[picked]={} rem[picked_part]={}",
        feasible[picked],
        rem[part_id[picked]],
    )

    # Apply selection
    new_selected_mask = jax.lax.cond(
        did_pick,
        lambda m: m.at[picked].set(True),
        lambda m: m,
        selected_mask,
    )

    picked_random = did_pick & use_fallback

    dbg(
        "FINAL: did_pick={} picked_random={} new_selected_count={}",
        did_pick,
        picked_random,
        jnp.sum(new_selected_mask.astype(jnp.int32)),
    )
    dbg("END SELECT STEP -------------------------")

    return key, new_selected_mask, did_pick, picked, picked_random


# @jax.jit
# def select_one_partition_matroid_toprelu(key, selected_mask, part_id, budget, proxy):
#     used, rem, cnt = _partition_accounting(selected_mask, part_id, budget)

#     rem_i = rem[part_id]
#     feasible = (~selected_mask) & (rem_i > 0)

#     sort_idx, part_s, feas_s, short_s, has_pos, scores = _shortlist_sorted(part_id, rem, proxy, feasible)
    
#     use_fallback = (~has_pos) & jnp.any(feasible)

#     # pick mask in SORTED space
#     pick_s = jnp.where(use_fallback, feas_s, short_s)

#     key, did_pick, picked_s = _sample_uniform_from_mask_sorted(key, pick_s)

#     picked = sort_idx[picked_s]  # map back to original index

#     new_selected_mask = jax.lax.cond(
#         did_pick,
#         lambda m: m.at[picked].set(True),
#         lambda m: m,
#         selected_mask,
#     )

#     picked_random = did_pick & use_fallback
#     return key, new_selected_mask, did_pick, picked_random


def _wall_time():
    return np.array(time.monotonic(), dtype=np.float64)

def _wall_time_dep(_dep):
    return np.array(time.monotonic(), dtype=np.float64)

def _jax_now():
    return jax.experimental.io_callback(
        _wall_time,
        jnp.zeros((), dtype=jnp.float64),
        ordered=True,
    )

def _jax_now_after(dep):
    """Like _jax_now but forced to run after dep is ready."""
    flat = jax.tree.leaves(dep)
    s = sum(jnp.sum(x) for x in flat) if flat else jnp.float64(0)
    return jax.experimental.io_callback(
        _wall_time_dep,
        jnp.zeros((), dtype=jnp.float64),
        s,
        ordered=True,
    )

def _tic():
    return _jax_now()

def _toc(label, start, dep=None):
    end = _jax_now_after(dep) if dep is not None else _jax_now()
    ms = (end - start) * 1000
    jax.debug.print("{} | {} ms", label, ms)
    return ms

@partial(jax.jit, static_argnames=("max_iters"))
def joint_subsel(rng, part_id, budget, scores, apdg_lr, interaction_matrix, prev_utils, max_iters):
    rng, key = jax.random.split(rng)
    Bs = part_id.shape[0]
    weights = jnp.zeros((Bs,), dtype=jnp.float32)
    selected_mask = jnp.zeros((Bs,), dtype=jnp.bool_)
    iters = jnp.sum(budget)
    picked_idxs = -jnp.ones((Bs,), dtype=jnp.int32)
    
    refit_fn = partial(apgd_partitions, (scores, interaction_matrix), part_id, max_iters, apdg_lr)
    grad_fn = partial(grad_utility, (scores, interaction_matrix))
    utilities = -jnp.inf*jnp.ones((Bs,max_iters), dtype=jnp.float32)

    def omp_step(key, weights, selected_mask, part_id, budget):
        # weights = jnp.where(selected_mask, weights, 0.0)
        proxy = grad_fn(weights)  # [N]

        key, selected_mask, did_pick, picked, picked_random = select_one_partition_matroid_toprelu(
            key, selected_mask, part_id, budget, proxy
        )

        _util = -jnp.inf*jnp.ones((max_iters,), dtype=jnp.float32) 

        weights, utility = jax.lax.cond(
            did_pick,
            lambda w: refit_fn(selected_mask, key, random_init=True),
            lambda w: (w, _util),
            weights,
        )

        return key, weights, utility, selected_mask, did_pick, picked, picked_random

    def body(i, state):
        key, weights, utilities, selected_mask, n_random, n_opt, picked_idxs = state
        key, weights, utility, selected_mask, did_pick, picked, picked_random = omp_step(
            key, weights, selected_mask, part_id, budget
        )
        picked_idxs = picked_idxs.at[i].set(picked)
        utilities = utilities.at[i].set(utility)

        # Accumulate counts
        n_random = n_random + picked_random.astype(jnp.int32)
        n_opt = n_opt + (did_pick & (~picked_random)).astype(jnp.int32)

        return key, weights, utilities, selected_mask, n_random, n_opt, picked_idxs

    init_state = (key, weights, utilities, selected_mask, jnp.int32(0), jnp.int32(0), picked_idxs)
    _, weights, utilities, mask, n_random, n_opt, picked_idxs = jax.lax.fori_loop(0, iters, body, init_state)

    jax.debug.print("joint_subsel done: picked_random={} picked_opt={}", n_random, n_opt)

    # t = jax.tree.map(lambda x, y: (x,y), part_id, picked_idxs, is_leaf=)
    jax.debug.print("partition {}", part_id)
    jax.debug.print("selected order {}", picked_idxs)
    jax.debug.print("bs/iters {} {}", Bs, iters)
    return mask, utilities, weights


@partial(jax.jit, static_argnames=("random_init", "max_iterations"))
def apgd_partitions(
    A: jnp.ndarray,            # [n, n]
    part_id: jnp.ndarray,      # [n]
    max_iterations: int, 
    learning_rate: float,
    selected_mask: jnp.ndarray,# [n] bool
    rng,
    random_init=False
):
    def project(w):
      """
        project over support
        w >= 0
        w_i = 0 if selected_mask[i] == False
      """
      w = jnp.where(selected_mask, w, 0.0)
      return jnp.maximum(w, 0.0)
      # return w
    w_init = jnp.zeros(part_id.shape, dtype=jnp.float32)
    if random_init:
      w_init = jax.random.uniform(rng, shape=part_id.shape)
    w = project(w_init)
    utility = -jnp.inf*jnp.ones(max_iterations, dtype=jnp.float32)
    y = w
    t = 1.0

    def body(iter, state):
        w, y, t, utility = state
        g = grad_utility(A, y)

        # ASCENT
        w_next = project(y + learning_rate * g)

        t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        y_next = w_next + ((t - 1.0) / t_next) * (w_next - w)
        _util = utility_fn(A, w_next)
        # jax.debug.print("inside util {}", _util)
        utility_new = utility.at[iter].set(_util)

        return (w_next, y_next, t_next, utility_new)

    w, _, _, utility = jax.lax.fori_loop(
        0, max_iterations, body, (w, y, t, utility)
    )

    return w, utility

def utility_fn(A, w):
    '''
    utility for greats obejective wrt w
    scores: [bs_train]
    interaction: [bs_train, bs_train]
    w: [bs_train]
    '''
    scores, interaction_matrix = A
    Bs = scores.shape[0]
    interaction = w*(interaction_matrix@w)
    chex.assert_shape(interaction, (Bs,))
    utility = jnp.sum(scores*w) - 0.5*jnp.sum(interaction)
    # DONOT mask non support entries
    # since support of w need not != support of g
    return utility

def grad_utility(A, w):
    '''
    gradient for greats obejective wrt w
    '''
    scores, interaction_matrix = A
    # g = scores - interaction_matrix@w
    g = scores - (interaction_matrix@w)
    # DONOT mask non support entries
    # since support of w need not != support of g
    return g



@partial(jax.jit, static_argnames=("ratio"))
def conflicting(grads, idxs, ratio):
  chex.assert_shape(idxs, (ratio,))
  # This is a hack, just to satisfy the compiler
  # idxs is already of shape [ratio]
  # as asserted above
  idxs = idxs[:ratio]
  grads = pack_pytree(grads)
  grads = grads[idxs]
  kernel = gram_linear(grads, grads)
  n = kernel.shape[0]
  kernel = kernel*(1 - jnp.eye(n,n))
  pairs = (jnp.sum(kernel < 0))/2
  # agree = (jnp.sum(kernel > 0) - n)/2
  return pairs

def _adjust_to_target(bud, cnt, target, key):
  # bud, cnt: int32 [C]
  # target: scalar int32
  init_diff = target - jnp.sum(bud)
  big = jnp.float32(1e9)

  def cond_fun(state):
    bud, diff, key = state
    return diff != 0

  def valid_idx(subkey, mask, bud):
    u = jax.random.uniform(subkey, bud.shape)
    u_masked = jnp.where(mask, u, big)
    idx = jnp.argmin(u_masked)
    return idx

  def body_fun(state):
    bud, diff, key = state
    key, subkey = jax.random.split(key)
    big = jnp.float32(1e9)

    def add_step(_):
      mask = bud < cnt  # can still increase
      any_valid = jnp.any(mask)

      def do_add(_):
        idx = valid_idx(subkey, mask, bud)
        bud2 = bud.at[idx].add(1)
        diff2 = diff - 1
        return bud2, diff2, key

      def no_add(_):
        # no valid positions; stop by zeroing diff
        return bud, jnp.int32(0), key

      return jax.lax.cond(any_valid, do_add, no_add, operand=None)

    def sub_step(_):
      mask = bud > 0  # can still decrease
      any_valid = jnp.any(mask)

      def do_sub(_):
        idx = valid_idx(subkey, mask, bud)
        bud2 = bud.at[idx].add(-1)
        diff2 = diff + 1
        return bud2, diff2, key

      def no_sub(_):
        return bud, jnp.int32(0), key

      return jax.lax.cond(any_valid, do_sub, no_sub, operand=None)

    return jax.lax.cond(diff > 0, add_step, sub_step, operand=None)

  init_state = (bud, init_diff.astype(jnp.int32), key)
  final_bud, final_diff, final_key = jax.lax.while_loop(cond_fun, body_fun, init_state)
  return final_bud


def standardize(x, axis=None):
  """
  Standardize tensor x along the given axis.
  If std == 0, outputs zeros on those entries instead of dividing.
  """
  mean = jnp.mean(x, axis=axis, keepdims=True)
  var = jnp.var(x, axis=axis, keepdims=True)
  std = jnp.sqrt(var)

  # Where std > 0, use (x-mean)/std; otherwise return 0
  standardized = jnp.where(std > 0, (x - mean) / std, 0.0)
  return standardized


def curr_budget(task_cfg, lamb, part_id, budget, gain, ratio, key):
  if not task_cfg["curricullum"]["enabled"]: return budget
  P = budget.shape[0]
  cnt = jnp.bincount(part_id, length=P).astype(jnp.int32)

  tot_gain = jnp.sum(gain)
  C = gain.shape[0]
  term = standardize(gain, axis=0)
  new_bud = ratio / C - (ratio / lamb) * term  # float [C]
  jax.debug.print("debug sum {}", jnp.sum(new_bud))
  # per-class bounds
  new_bud = jnp.clip(new_bud, 0.0, cnt.astype(new_bud.dtype))
  new_bud = jnp.rint(new_bud).astype(jnp.int32)
  adap_bud = new_bud

  # global target, clamped to capacity
  total_cap = jnp.sum(cnt)
  target = jnp.round(ratio).astype(jnp.int32)
  target = jnp.clip(target, 0, total_cap)

  # random, exact adjustment under jit
  new_bud = _adjust_to_target(new_bud, cnt, target, key)
  jax.debug.print("orig_bud {} {}", budget, jnp.sum(budget))
  jax.debug.print("adap_bud {} {}", adap_bud, jnp.sum(adap_bud))
  jax.debug.print("new_bud {} {}", new_bud, jnp.sum(new_bud))
  return new_bud


def per_class_utilities(grads, anchors, src_v, num_train_sources, lr1, lr2, interaction_matrix, weights):
    # sims: [N, A]
    sims = gram_linear(grads, anchors).astype(jnp.float32)

    # one_hot: [A, C]
    one_hot = jax.nn.one_hot(src_v, num_classes=num_train_sources, dtype=sims.dtype)

    # masked mean over anchors per class
    sims_exp = sims[:, :, None]           # [N, A, 1]
    mask = one_hot[None, :, :]           # [1, A, C]

    num = jnp.sum(sims_exp * mask, axis=1)      # [N, C]
    denom = jnp.sum(mask, axis=1)               # [1, C]
    denom = jnp.clip(denom, 1.0)                # avoid div by zero
    scores_per_class = num / denom              # [N, C]

    def util_for_class(scores_c):
        # ================================================================
        # IMPORTANT: keep this in sync with the weighted PartitionSel
        # objective used by joint_subsel.
        #
        #   U(w) = <w, lr * score + 0.5 * lr^2 * diag(K)>
        #          - 0.5 * w^T (lr^2 * K) w
        #
        # Do NOT drop diag(K). The diagonal is the self-curvature term
        # ||g_i||^2 that keeps the learned prototype weights finite.
        # This function feeds the per-class curriculum bookkeeping, so it
        # must measure the same utility that the joint weighted selector uses.
        # ================================================================
        weighted_scores_c = lr1 * scores_c + 0.5 * lr2 * jnp.diag(interaction_matrix)
        weighted_interaction = lr2 * interaction_matrix
        return utility_fn((weighted_scores_c, weighted_interaction), weights)

    # scores_per_class: [N, C], vmap over classes (axis=1)
    final_utils = jax.vmap(util_for_class, in_axes=1, out_axes=0)(scores_per_class)  # [C]
    return final_utils


@partial(nnx.jit, static_argnames=("out_len"))
def greats_selection(scores, interaction_matrix, out_len: int, source_mask=None, limit=None):
  """
  scores: (n,) 1D array
  interaction_matrix: (n, n)
  out_len: int
  returns: (K,) int32 selected indices
  """
  if limit is None: limit = out_len
  W = interaction_matrix
  if source_mask is not None:
    chex.assert_equal_shape([scores, source_mask])
    chex.assert_type(source_mask, jnp.bool)
    scores = jnp.where(source_mask, scores, -jnp.inf)

  effective_k = jnp.minimum(out_len, jnp.where(limit is None, out_len, limit))
  # Buffers (static shapes)
  selected0 = -jnp.ones((out_len,), dtype=jnp.int32)  # filled prefix; unused stay -1

  def body(i, state):
    cur_scores, selected = state
    idx = jnp.argmax(cur_scores)                      # int32
    selected = selected.at[i].set(idx)
    cur_scores = cur_scores - W[idx, :]  # subtract interactions
    cur_scores = cur_scores.at[idx].set(-jnp.inf)       # prevent reselection
    return (cur_scores, selected)

  final_scores, selected = jax.lax.fori_loop(0, effective_k, body, (scores, selected0))
  return selected  # [k], with first `effective_k` filled, rest = -1


# @partial(nnx.jit, static_argnums=(2,))
# def greats_selection(scores, interaction_matrix, K: int):
#   """
#   scores: (n,) 1D array
#   interaction_matrix: (n, n)
#   K: int
#   returns: (K,) int32 selected indices
#   """
#   W = interaction_matrix

#   # preallocate output
#   selected0 = jnp.full((K,), -1, dtype=jnp.int32)

#   def body(i, state):
#     cur_scores, selected = state
#     idx = jnp.argmax(cur_scores)                      # int32
#     selected = selected.at[i].set(idx)
#     cur_scores = cur_scores - W[idx, :]  # subtract interactions
#     cur_scores = cur_scores.at[idx].set(-jnp.inf)       # prevent reselection
#     return (cur_scores, selected)

#   final_scores, selected = jax.lax.fori_loop(0, K, body, (scores, selected0))
#   return selected

@partial(jax.jit, static_argnums=(1,))
def facility_location_old2(S: jnp.ndarray, k: int) -> jnp.ndarray:
  """
  Args:
      S: [n, m] similarity matrix. Objective is sum_i max_{j in A} S[i, j].
      k: number of items to select (must be <= n).

  Returns:
      idxs: [k] int32 array of selected indices.
  """
  S = jnp.transpose(S) # (m,n) -> thus n is axis where points are choosen from
  m, n = S.shape
  # k = jnp.minimum(k, n)

  # best[i] = best similarity achieved so far for row i (coverage)
  # best = jnp.full((n,), -jnp.inf, dtype=S.dtype)
  # best = jnp.zeros((n,), dtype=S.dtype)
  # selected_mask = jnp.zeros((n,), dtype=bool)
  # out = jnp.full((k,), -1, dtype=jnp.int32)
  gains0 = S.sum(axis=0)
  j0 = jnp.argmax(gains0) # choosing axis = 1
  best = S[:, j0]   # best[i] = max_{j in {j0}} S[i, j]

  # bookkeeping
  selected_mask = jnp.zeros((n,), dtype=bool).at[j0].set(True)
  out = jnp.full((k,), -1, dtype=jnp.int32).at[0].set(j0)


  def step(t, carry):
    best, selected_mask, out = carry
    # Marginal gain for adding column j: sum_i max(0, S[i,j] - best[i])
    gains = jnp.maximum(S - best[:, None], 0.0).sum(axis=0)
    # Mask already-selected candidates
    gains = jnp.where(selected_mask, -jnp.inf, gains)

    j = jnp.argmax(gains)  # tie-breaks to smallest index
    # Update coverage and bookkeeping
    best = jnp.maximum(best, S[:, j])
    selected_mask = selected_mask.at[j].set(True)
    out = out.at[t].set(jnp.int32(j))
    return (best, selected_mask, out)

  best, selected_mask, out = jax.lax.fori_loop(1, k, step, (best, selected_mask, out))
  jax.debug.print("selected {}", out)
  return out

@partial(jax.jit, static_argnums=(1,))  # match greedy_fairot: k, reg, iters static
def facility_location(
    S: jnp.ndarray,                 # [n, n] similarities
    k,                         # capacity of output array (static length)
    src_mask: jnp.ndarray = None,   # [n] bool: eligible columns (candidates)
    tgt_mask: jnp.ndarray = None,   # [n] bool: rows that contribute to coverage
    limit: jnp.ndarray | None = None,  # dynamic effective k for this call
) -> jnp.ndarray:
  # S = (1+S)/2
  """
  Facility-Location greedy with early stop and padding:
    - Returns int32 array of length k.
    - Fills only the first `effective_k` positions; the rest remain -1.
    - `effective_k = min(k, limit, #eligible)`; if no eligible, returns all -1.
    - Masks: src_mask filters candidate columns; tgt_mask filters covered rows.
  """
  n = S.shape[0]

  # default masks
  src_mask = jnp.ones((n,), dtype=bool) if src_mask is None else src_mask
  tgt_mask = jnp.ones((n,), dtype=bool) if tgt_mask is None else tgt_mask

  # bound effective steps by k, limit (if given), and #eligible candidates
  max_possible = jnp.sum(src_mask.astype(jnp.int32))
  base_k = k if limit is None else jnp.minimum(k, limit)
  effective_k = jnp.minimum(base_k, max_possible)

  # static-shaped buffers
  selected = -jnp.ones((k,), dtype=jnp.int32)
  chosen   = jnp.zeros((n,), dtype=bool)
  neg_inf = -jnp.inf
  S_clean = jnp.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
  # early exit: nothing to pick
  def early_return(_):
    return selected
  def proceed(_):
    # bootstrap: pick first column by masked column-sum over covered rows
    # gains0[j] = sum_{i in tgt_mask} S[i, j], masked by src eligibility
    gains0 = (S_clean * tgt_mask[:, None]).sum(axis=0)
    gains0 = jnp.where(src_mask, gains0, neg_inf)
    j0 = jnp.argmax(gains0)

    selected0 = selected.at[0].set(j0)
    chosen0   = chosen.at[j0].set(True)
    # coverage vector over rows; we’ll always mask rows via tgt_mask in deltas
    best0 = S[:, j0]

    # loop state
    state0 = (best0, chosen0, selected0, 1)

    def cond_fun(state):
      _, _, _, t = state
      return t < effective_k

    def body_fun(state):
      best, chosen, selected, t = state
      # marginal gains: sum_i max(0, S[i,j] - best[i]) only over tgt_mask rows
      delta = S_clean - best[:, None]
      delta = jnp.where(tgt_mask[:, None], delta, neg_inf)  # masked rows contribute 0 after relu
      delta = jnp.where(src_mask[None, :], delta, neg_inf)  # masked cols contribute 0 after relu
      gains = jnp.maximum(delta, 0.0).sum(axis=0)
      # block already chosen and ineligible columns
      valid = src_mask & (~chosen)
      gains = jnp.where(valid, gains, neg_inf)

      j = jnp.argmax(gains)
      # update coverage and bookkeeping
      best = jnp.maximum(best, S[:, j])
      best = jnp.where(tgt_mask, best, neg_inf)
      chosen  = chosen.at[j].set(True)
      selected = selected.at[t].set(jnp.int32(j))
      t = t + 1
      return (best, chosen, selected, t)

    _, _, selected_fin, _ = jax.lax.while_loop(cond_fun, body_fun, state0)
    return selected_fin

  # If effective_k == 0, return all -1; else run greedy.
  idx = jax.lax.cond(effective_k == 0, early_return, proceed, operand=None)
  # jax.debug.print("selected {} {}", effective_k, idx)
  return idx





def stable_entropy(gamma: jnp.ndarray) -> float:
    mask = gamma > 0
    # return -np.sum(gamma[mask] * np.log(np.maximum(gamma[mask], 1e-12)))
    return -jnp.sum(gamma * jnp.log(jnp.maximum(gamma, 1e-12)))


@partial(jax.jit, static_argnums=(2, 3,4))  # reg, iters are static; k/limit can vary per class
def greedy_fairot(
    S: jnp.ndarray,                 # [n,m] similarities
    dist: jnp.ndarray,              # [n,m] distances (or cost terms)
    k: int,                         # capacity of output array (static length)
    reg: float,
    iters: int,
    src_mask: jnp.ndarray = None,   # [n] bool (optional)
    tgt_mask: jnp.ndarray = None,   # [m] bool (optional)
    limit: jnp.ndarray | None = None,  # dynamic effective k for this call
) -> jnp.ndarray:
    """
    Greedy selection with early stop:
      - Returns int32 array of length k.
      - Fills only the first `effective_k` positions; the rest remain -1.
      - `effective_k = min(k, limit)` if limit is given, else `k`.
    """
    n, m = S.shape

    # Effective steps to run (dynamic per class)
    effective_k = jnp.minimum(k, jnp.where(limit is None, k, limit))

    # Buffers (static shapes)
    selected = -jnp.ones((k,), dtype=jnp.int32)  # filled prefix; unused stay -1
    chosen   = jnp.zeros((n,), dtype=bool)       # boolean mask
    t        = jnp.int32(0)                      # how many selected so far

    # Optional masks just pass through to exact_gain; if None, pretend True
    # (If your exact_gain handles None, you can drop these two lines.)
    src_mask = jnp.ones((n,), dtype=bool) if src_mask is None else src_mask
    tgt_mask = jnp.ones((m,), dtype=bool) if tgt_mask is None else tgt_mask

    def body_fun(state):
        selected, chosen, t = state

        candidates = jnp.arange(n, dtype=jnp.int32)

        def gain_of(idx):
            return exact_gain(
                selected, idx, S, dist, k, reg, iters,
                src_mask=src_mask, tgt_mask=tgt_mask
            )  # scalar

        gains = jax.vmap(gain_of)(candidates)          # [n]
        gains = jnp.where(chosen, -jnp.inf, gains)     # mask already chosen
        best = jnp.argmax(gains)                      # int32

        selected = selected.at[t].set(best)
        chosen = chosen.at[best].set(True)
        t = t + jnp.int32(1)
        return (selected, chosen, t)

    def cond_fun(state):
        # keep iterating while we still need to pick more
        _, _, t = state
        return t < effective_k

    # Run only as many steps as needed; shape stays static.
    selected, _, _ = jax.lax.fori_loop(0, effective_k, lambda i, state: body_fun(state), (selected, chosen, t))
    return selected  # [k], with first `effective_k` filled, rest = -1


def _row_slice(x, i):
    # Returns x[i:i+1, :] with static shape (1, x.shape[1])
    i = jnp.asarray(i, dtype=jnp.int32)
    return jax.lax.dynamic_slice(x, (i, 0), (1, x.shape[1]))

@partial(jax.jit, static_argnums=(4,5,6))
def exact_gain(
    P, new_idx, S, D, k, reg, iters, src_mask, tgt_mask
):
    
    P_idx = jnp.array(P, dtype=jnp.int32)
    P_empty = (P_idx.size == 0)
    
    if P_empty:
        S_P_new = _row_slice(S, new_idx)                           # (1, n)
        D_P_new = _row_slice(D, new_idx)    # (1, n)
        mask_new = None
        if(src_mask is not None):
            mask_new = jax.lax.dynamic_slice_in_dim(src_mask, new_idx, 1)
        _, obj_new = pot_jax_masked(S_P_new, k, reg, D_P_new, iters=iters, src_mask=mask_new, tgt_mask=tgt_mask)
        # _, obj_new = pot_jax(S_P_new, k, reg, D_P_new, iters=iters)
        obj_old = jnp.array(0.0, dtype=S.dtype)
        return (obj_new - obj_old).astype(S.dtype)
    else:
        # Old set
        S_P_old = S[P_idx, :] # (|P|, n) — |P| static within a jit/vmap call
        D_P_old = D[P_idx, :]
        mask_old = None
        if(src_mask is not None):
            mask_old = src_mask[P_idx]

        # New set: append the single row v with static (1,n) slice
        S_P_new = jnp.concatenate([S_P_old, _row_slice(S, new_idx)], axis=0)   # (|P|+1, n)
        D_P_new = jnp.concatenate([D_P_old, _row_slice(D, new_idx)], axis=0)
        mask_new = None
        if(src_mask is not None):
            mask_new = jnp.concatenate([mask_old, jax.lax.dynamic_slice_in_dim(src_mask, new_idx, 1)], axis=0)
        
        _, obj_new = pot_jax_masked(S_P_new, k, reg, D_P_new, iters=iters, src_mask=mask_new, tgt_mask=tgt_mask)
        _, obj_old = pot_jax_masked(S_P_old, k, reg, D_P_old, iters=iters, src_mask=mask_old, tgt_mask=tgt_mask)
        return (obj_new - obj_old).astype(S.dtype)
        


@partial(jax.jit, static_argnums=(3,6,7),  donate_argnums=(2,))
def entropic_partial_wasserstein_masked(
    a: jnp.ndarray,                 # [m] >=0
    b: jnp.ndarray,                 # [n] >=0
    cost: jnp.ndarray,              # [m,n]
    reg: float,                     # >0
    m: float,                       # total mass to transport
    mask2d: jnp.ndarray,            # [m,n] bool, True where valid
    max_iters: int = 1000,
    stop_thr: float = 1e-12,
):
    """
    Partial OT with entropic regularization via Bregman-Dykstra, respecting masks.
    Constraints: (row) K 1 <= a, (col) K^T 1 <= b, (mass) 1^T K 1 = m, K>=0,
    and K_ij=0 for masked-out pairs.
    """
    # a, b, C, M_c, scatter_back = _compact(a, b, cost, mask2d)
    dtype = cost.dtype
    eps   = jnp.asarray(1e-12, dtype)
    one   = jnp.asarray(1.0, dtype)

    # Initialize only on valid entries
    negC_over_reg = -cost / reg
    logK = jnp.where(mask2d, negC_over_reg, -jnp.inf)
    lse  = jax.scipy.special.logsumexp(logK)         # scalar over all valid entries
    K    = m * jnp.exp(logK - lse)                   # zeros on invalid by construction

    # Dykstra multiplicative residuals (keep 1.0 on invalid entries)
    r_row  = jnp.ones_like(K)
    r_col  = jnp.ones_like(K)
    r_mass = jnp.ones_like(K)

    def body_fn(state):
        K, r_row, r_col, r_mass, _, it, _ = state
        K_prev = K

        # 1) Project rows: K 1 <= a (auto-equality when mass constraint active)
        K_row    = K * r_row
        row_sums = jnp.sum(K_row, axis=1) + eps
        row_scale = jnp.minimum(a / row_sums, one)                  # [m]
        K_after_rows = K_row * row_scale[:, None]
        upd_r_row = r_row * (K / (K_after_rows + eps))
        r_row = jnp.where(mask2d, upd_r_row, 1.0)

        # 2) Project cols: K^T 1 <= b
        K_col    = K_after_rows * r_col
        col_sums = jnp.sum(K_col, axis=0) + eps
        col_scale = jnp.minimum(b / col_sums, one)                  # [n]
        K_after_cols = K_col * col_scale[None, :]
        upd_r_col = r_col * (K_after_rows / (K_after_cols + eps))
        r_col = jnp.where(mask2d, upd_r_col, 1.0)

        # 3) Project mass: 1^T K 1 = m
        K_mass = K_after_cols * r_mass
        total  = jnp.sum(K_mass) + eps
        K_new  = K_mass * (m / total)
        upd_r_mass = r_mass * (K_after_cols / (K_new + eps))
        r_mass = jnp.where(mask2d, upd_r_mass, 1.0)

        # keep masked cells identically zero
        K_new = jnp.where(mask2d, K_new, 0.0)

        err = jnp.linalg.norm(K_new - K_prev)
        bad = jnp.any(~jnp.isfinite(K_new))

        return (K_new, r_row, r_col, r_mass, err, it + 1, bad)

    def cond_fn(state):
        _, _, _, _, err, it, bad = state
        return (err > stop_thr) & (it < max_iters) & (~bad)

    init = (K, r_row, r_col, r_mass,
            jnp.asarray(jnp.inf, dtype), jnp.array(0, jnp.int32), jnp.array(False))
    K, r_row, r_col, r_mass, err, it, bad = jax.lax.while_loop(cond_fn, body_fn, init)

    # # Optional warning (printed only if needed)
    # jax.lax.cond(bad | (it >= max_iters),
    #              lambda _: jax.debug.print("EPOT warn: it={} err={}", it, err),
    #              lambda _: None, operand=None)

    return K  # [m,n], zero on masked entries

@partial(jax.jit, static_argnums=(1,2,4))
def pot_jax_masked(
    S_sub: jnp.ndarray,             # [m,n] similarity (for objective)
    k: int,
    reg: float,
    D_sub: jnp.ndarray,             # [m,n] cost for transport
    iters: int = 20,
    src_mask: jnp.ndarray = None,   # [m] bool
    tgt_mask: jnp.ndarray = None,   # [n] bool
    mask2d_sub: jnp.ndarray = None
):
    m, n = S_sub.shape
    if src_mask is None:
        src_mask = jnp.ones((m,), dtype=bool)
    if tgt_mask is None:
        tgt_mask = jnp.ones((n,), dtype=bool)

    # valid pairs only
    if mask2d_sub is None:
        mask2d = src_mask[:, None] & tgt_mask[None, :]
    else: mask2d = mask2d_sub

    # source marginal: 1 for valid sources, 0 otherwise  (sum = #valid_src)
    a = jnp.where(src_mask, 1.0, 0.0).astype(D_sub.dtype)

    # target marginal: spread k uniformly over valid targets (sum = k)
    n_t = jnp.maximum(jnp.sum(tgt_mask), 1)  # avoid div by zero
    b_unit = (k / n_t).astype(D_sub.dtype)
    b = jnp.where(tgt_mask, b_unit, 0.0)

    # mass to transport limited by feasible active mass
    m_active = jnp.minimum(jnp.sum(a), jnp.sum(b))

    gamma = entropic_partial_wasserstein_masked(
        a=a, b=b, cost=D_sub, reg=reg, m=m_active, mask2d=mask2d,
        max_iters=iters, stop_thr=1e-12
    )

    # objective on valid cells only
    obj = jnp.sum((S_sub * gamma) * mask2d) + reg * stable_entropy(gamma)
    return gamma, obj



