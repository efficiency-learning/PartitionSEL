# Re-export public API for backward compatibility.
# Callers can continue using `from tunix.examples.data import ift_dataset as data_lib_ift`
# and access `data_lib_ift.create_datasets(...)` etc.

from .configs import (
    LAWINSTRUCT_DICT,
    LEGALBENCH_SOURCE_MODE,
    LEGALBENCH_TASK_TYPE_MAP,
    LEGALBENCH_TASKS,
    METAMATH_DICT,
    MOL_DICT,
    TIGER_DICT,
)
from .create_datasets import create_datasets
from .datasets import (
    convert_to_chatformat,
    get_colm,
    get_dolma,
    get_eval_greats,
    get_lawinstruct,
    get_legalbench,
    process_data,
)
from .loaders import InfiniteLoader, make_jax_collate, make_seeded_loader
from .source_masks import (
    get_source_masks,
    scatter_active_to_full,
)
from .utils import (
    debug,
    filter_all_ignored_labels,
    get_splits,
    get_subjects,
    repeat_dataset,
    split_by_group,
    split_data,
    to_np_int32,
)
