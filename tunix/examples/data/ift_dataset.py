"""Backward-compatible wrapper — all code now lives in tunix.examples.data.ift/."""

# Re-export the full public API so existing callers are unchanged.
from tunix.examples.data.ift import (  # noqa: F401
    # configs
    LAWINSTRUCT_DICT,
    LEGALBENCH_SOURCE_MODE,
    LEGALBENCH_TASK_TYPE_MAP,
    LEGALBENCH_TASKS,
    METAMATH_DICT,
    MOL_DICT,
    TIGER_DICT,
    # create_datasets
    create_datasets,
    # datasets
    convert_to_chatformat,
    get_colm,
    get_eval_greats,
    get_lawinstruct,
    get_legalbench,
    process_data,
    # loaders
    InfiniteLoader,
    make_jax_collate,
    make_seeded_loader,
    # source_masks
    get_source_masks,
    scatter_active_to_full,
    # utils
    debug,
    filter_all_ignored_labels,
    get_splits,
    get_subjects,
    repeat_dataset,
    split_by_group,
    split_data,
    to_np_int32,
)
