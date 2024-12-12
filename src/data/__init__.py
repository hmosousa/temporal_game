from typing import Any, Dict, Tuple

from torch.utils.data import Dataset

from src.data.levels import load_levels

from src.data.q_timelines import load_qtimelines
from src.data.small_timelines import load_small_timelines
from src.data.timeset import load_timeset
from src.data.utils import balance_dataset_classes

DATASETS = {
    "q_timelines": load_qtimelines,
    "timeset": load_timeset,
    "levels": load_levels,
}


def load_dataset(
    dataset_name: str, split: str, config: Dict[str, Any] = {}
) -> Tuple[Dataset, Dataset]:
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} not found. Valid datasets are: {list(DATASETS.keys())}"
        )
    return DATASETS[dataset_name](split=split, **config)


__all__ = [
    "load_dataset",
    "load_qtimelines",
    "load_small_timelines",
    "load_timeset",
    "load_levels",
    "balance_dataset_classes",
]
