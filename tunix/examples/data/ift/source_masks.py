import numpy as np


def scatter_active_to_full(active_vals, classes_active, MC, dtype=np.int32):
    """
    Convert a compact "active-class" vector into a full vector indexed by class id.
    """
    active_vals = np.asarray(active_vals, dtype=dtype)
    classes_active = np.asarray(classes_active, dtype=np.int32)

    assert classes_active.ndim == 1, "classes_active must be 1D."
    assert active_vals.ndim == 1, "active_vals must be 1D."
    assert active_vals.shape[0] == classes_active.shape[0], "active_vals and classes_active must align."
    assert int(MC) > 0, "MC must be positive."
    assert classes_active.min(initial=0) >= 0 and classes_active.max(initial=-1) < int(MC), (
        "classes_active must be in [0, MC)."
    )
    assert np.unique(classes_active).size == classes_active.size, "classes_active must be unique."

    full = np.zeros((int(MC),), dtype=dtype)
    full[classes_active] = active_vals
    return full


def _increase_array_to_threshold_masked(arr, threshold, eligible_mask):
    """
    Masked version of increase_array_to_threshold with identical behavior on the eligible subset.
    """
    values = np.asarray(arr, dtype=np.int32).copy()
    eligible_mask = np.asarray(eligible_mask, dtype=bool)

    assert values.ndim == 1 and eligible_mask.ndim == 1
    assert values.shape[0] == eligible_mask.shape[0]
    assert np.all(values >= 0), "arr must be nonnegative."

    need = int(threshold) - int(values.sum())
    if need < 0:
        raise ValueError("threshold must be >= current sum")
    if need == 0:
        return values

    eligible_idx = np.where(eligible_mask)[0]
    if eligible_idx.size == 0:
        raise ValueError("No eligible indices to increase but threshold requires increases.")

    order_local = np.lexsort((eligible_idx, values[eligible_idx]))  # indices into eligible_idx
    order = eligible_idx[order_local]                               # actual indices in [0..len(values)-1]

    n = int(order.size)
    for step in range(need):
        i = order[step % n]
        values[i] += 1
    return values


def _validate_inputs_and_active_view(
    src, N, MC, B,
    uniq, classes_active, y, freqs_active,
    include_full,
    strategy, per_class_start,
):
    """
    Validates inputs + active-class view invariants + include_full feasibility.

    Returns:
      include_full_arr: [K] int32
      forced_active: [C_active] bool
      B_forced: int
      B_free: int
      N_free: int
    """
    assert src.ndim == 1 and src.shape[0] == N
    assert N > 0
    assert MC > 0
    assert 0 <= B <= N
    assert src.min() >= 0 and src.max() < MC

    C_active = int(classes_active.size)
    assert uniq.ndim == 1 and classes_active.ndim == 1
    assert y.ndim == 1 and y.shape == (N,)
    assert C_active >= 1
    assert np.all(uniq[:-1] < uniq[1:]), "uniq must be strictly increasing."
    assert np.all(classes_active[:-1] < classes_active[1:]), "classes_active must be strictly increasing."
    assert y.min() >= 0 and y.max() < C_active, "inv/y must be in [0, C_active)."
    assert np.array_equal(uniq[y].astype(src.dtype), src), "Reconstruction uniq[inv] != src."
    assert np.array_equal(classes_active[y].astype(src.dtype), src), "Reconstruction classes_active[y] != src."

    assert freqs_active.shape == (C_active,)
    assert int(freqs_active.sum()) == N, "Active class counts must sum to N."
    assert np.all(freqs_active > 0), "All active classes must have positive frequency."

    if strategy == "proportional":
        assert per_class_start in ("floor", "ceil"), "per_class_start must be 'floor' or 'ceil'."
    elif strategy == "none":
        pass
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    if include_full is None:
        include_full_arr = np.empty((0,), dtype=np.int32)
    else:
        include_full_arr = np.asarray(include_full, dtype=np.int32).reshape(-1)

    if include_full_arr.size > 0:
        assert include_full_arr.min() >= 0 and include_full_arr.max() < MC, "include_full must be in [0, MC)."
        assert np.unique(include_full_arr).size == include_full_arr.size, "include_full must be unique."

    forced_active = np.isin(classes_active, include_full_arr)
    B_forced = int(freqs_active[forced_active].sum())
    B_free = int(B - B_forced)
    assert B_free >= 0, f"include_full forces selecting {B_forced} items, which exceeds B={B}."

    N_free = int(freqs_active[~forced_active].sum())
    assert B_free <= N_free, f"Remaining budget B_free={B_free} exceeds remaining available N_free={N_free}."

    return include_full_arr, forced_active, B_forced, B_free, N_free


def _validate_packed_outputs(
    src, MC, B,
    classes_active, k_active,
    k_per_class_vec, out_offsets_vec,
):
    """
    Post-allocation logic checks: scatter correctness, availability, offsets correctness.
    """
    src = np.asarray(src, dtype=np.int32)
    classes_active = np.asarray(classes_active, dtype=np.int32)
    k_active = np.asarray(k_active, dtype=np.int32)
    k_per_class_vec = np.asarray(k_per_class_vec, dtype=np.int32)
    out_offsets_vec = np.asarray(out_offsets_vec, dtype=np.int32)

    C_active = int(classes_active.size)

    assert k_active.shape == (C_active,)
    assert k_per_class_vec.shape == (MC,)
    assert out_offsets_vec.shape == (MC,)

    assert np.all(k_active >= 0)
    assert np.all(k_per_class_vec >= 0)
    assert int(k_active.sum()) == int(B)
    assert int(k_per_class_vec.sum()) == int(B)

    assert np.array_equal(k_per_class_vec[classes_active], k_active), "Scatter mismatch: full[classes_active] != k_active."
    non_active = np.ones(MC, dtype=bool)
    non_active[classes_active] = False
    assert np.all(k_per_class_vec[non_active] == 0), "Non-active classes must have zero allocation."

    counts_by_class = np.bincount(src, minlength=MC).astype(np.int32)
    excess = k_per_class_vec - counts_by_class
    assert np.all(excess <= 0), (
        "Requested more than available in some real classes. "
        f"Max excess={int(excess.max())}."
    )

    prefix = np.zeros((MC,), dtype=np.int32)
    if MC > 1:
        prefix[1:] = np.cumsum(k_per_class_vec[:-1], dtype=np.int64).astype(np.int32)
    assert np.array_equal(out_offsets_vec, prefix), "out_offsets_vec must be prefix-sum of k_per_class_vec."

    assert np.all(out_offsets_vec >= 0) and np.all(out_offsets_vec <= B), "Offsets out of range."
    assert np.all(out_offsets_vec[:-1] <= out_offsets_vec[1:]), "Offsets must be non-decreasing."
    assert np.all(out_offsets_vec + k_per_class_vec <= B), "Class slice overruns global cap B."
    if MC > 0:
        assert int(out_offsets_vec[-1] + k_per_class_vec[-1]) == int(B), "Last class slice must end at B."


def get_source_masks(
    sources,            # [N] class ids; guaranteed to be in [0..MC-1]
    B: int,             # total number of items to select across all classes
    max_classes: int,   # MC; fixed universe of class ids 0..MC-1
    strategy: str = "proportional",
    per_class_start: str = "floor",
    include_full=None,  # list/array of real class ids to fully include (ignored for strategy="none")
):
    """
    Build per-class selection metadata.

    include_full (only meaningful for strategy="proportional"):
      Real class ids (0..MC-1). For these classes, we force selecting all examples
      from that class (k[c] = count_in_data[c]). Remaining budget is allocated to
      other classes according to `strategy`.
    """
    src = np.asarray(sources, dtype=np.int32)
    N = int(src.shape[0])
    MC = int(max_classes)
    B = int(B)

    if src.ndim != 1:
        raise ValueError(f"sources must be 1D, got shape={src.shape}.")
    if N <= 0:
        raise ValueError("sources must be non-empty.")
    if MC <= 0:
        raise ValueError("max_classes must be positive.")
    if B < 0 or B > N:
        raise ValueError(f"B must be in [0, N]. Got B={B}, N={N}.")
    if src.min() < 0 or src.max() >= MC:
        raise ValueError(f"sources must be in [0, {MC-1}] (got min={src.min()}, max={src.max()}).")

    uniq, inv = np.unique(src, return_inverse=True)
    classes_active = uniq.astype(np.int32)
    y = inv.astype(np.int32)
    C_active = int(classes_active.size)

    if C_active > MC:
        raise ValueError(f"max_classes={MC} < actual active classes={C_active}. Increase max_classes.")

    freqs_active = np.bincount(y, minlength=C_active).astype(np.int32)

    include_full_eff = None if strategy == "none" else include_full

    include_full_arr, forced_active, B_forced, B_free, N_free = _validate_inputs_and_active_view(
        src=src, N=N, MC=MC, B=B,
        uniq=uniq, classes_active=classes_active, y=y, freqs_active=freqs_active,
        include_full=include_full_eff,
        strategy=strategy, per_class_start=per_class_start,
    )

    if strategy == "none":
        num_per_class_active = np.int32([B])
        classes_active = np.array([0], dtype=np.int32)
        C_active = 1
        freqs_active = np.array([N], dtype=np.int32)
        y = np.zeros(N, dtype=np.int32)

        assert num_per_class_active.shape == (1,)
        assert int(num_per_class_active.sum()) == B

        k_active = np.minimum(num_per_class_active, freqs_active).astype(np.int32)
        assert int(k_active.sum()) == B

    elif strategy == "proportional":
        # Forced: take all available from forced classes (active space)
        k_forced_active = np.where(forced_active, freqs_active, 0).astype(np.int32)

        # Free proportional allocation over non-forced classes, total B_free
        raw_free = np.zeros_like(freqs_active, dtype=np.float64)
        if B_free > 0:
            denom = float(freqs_active[~forced_active].sum())
            assert denom > 0.0
            raw_free = (freqs_active.astype(np.float64) / denom) * float(B_free)
            raw_free = raw_free * (~forced_active).astype(np.float64)

        if per_class_start == "floor":
            num_free_active = np.floor(raw_free).astype(np.int32)
        else:
            num_free_active = np.ceil(raw_free).astype(np.int32)

        num_free_active = _increase_array_to_threshold_masked(
            num_free_active, B_free, eligible_mask=(~forced_active)
        )

        if B_free == 0:
            assert int(num_free_active.sum()) == 0, "B_free=0 but free allocation is nonzero."

        num_per_class_active = (k_forced_active + num_free_active).astype(np.int32)

        assert num_per_class_active.shape == (C_active,), "Allocation must be length C_active."
        assert np.issubdtype(num_per_class_active.dtype, np.integer)
        assert np.all(num_per_class_active >= 0), "Allocation must be nonnegative."
        assert int(num_per_class_active.sum()) == B, "Allocation must sum exactly to B."

        over = num_per_class_active - freqs_active
        assert np.all(over <= 0), (
            "No-redistribution mode violated: allocated more than available in some active classes. "
            f"Max overallocation={int(over.max())}."
        )

        k_active = np.minimum(num_per_class_active, freqs_active).astype(np.int32)

        assert int(k_active.sum()) == B, (
            "Availability capping reduced the total below B. "
            "Either adjust allocation to respect class counts or add redistribution."
        )
        assert np.all(k_active >= 0), "k_active must be nonnegative."
        assert k_active.shape[0] == classes_active.shape[0], "k_active must align with classes_active."

        if include_full_arr.size > 0:
            assert np.all(k_active[forced_active] == freqs_active[forced_active]), (
                "include_full classes must take all available examples."
            )

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    k_per_class_vec = scatter_active_to_full(k_active, classes_active, MC, dtype=np.int32)

    out_offsets_vec = np.zeros((MC,), dtype=np.int32)
    if MC > 1:
        out_offsets_vec[1:] = np.cumsum(k_per_class_vec[:-1], dtype=np.int64).astype(np.int32)

    _validate_packed_outputs(
        src=src, MC=MC, B=B,
        classes_active=classes_active, k_active=k_active,
        k_per_class_vec=k_per_class_vec, out_offsets_vec=out_offsets_vec,
    )

    k_per_class = np.repeat(k_per_class_vec[None, :], N, axis=0)
    out_offsets = np.repeat(out_offsets_vec[None, :], N, axis=0)

    meta = dict(
        k_eff=k_per_class,        # [N, MC]
        out_offsets=out_offsets,  # [N, MC]
        sources=src,
    )
    return meta
