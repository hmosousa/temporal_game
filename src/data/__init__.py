from typing import Any, Dict, Tuple

from torch.utils.data import Dataset

from src.data.q_timelines import load_qtimelines
from src.data.small_timelines import load_small_timelines
from src.data.utils import balance_dataset_classes

DATASETS = {
    "q_timelines": load_qtimelines,
}


def load_dataset(
    dataset_name: str, config: Dict[str, Any] = {}
) -> Tuple[Dataset, Dataset]:
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} not found. Valid datasets are: {list(DATASETS.keys())}"
        )
    trainset = DATASETS[dataset_name](split="train", **config)
    validset = DATASETS[dataset_name](split="valid", **config)
    return trainset, validset


__all__ = [
    "load_dataset",
    "load_qtimelines",
    "load_small_timelines",
    "balance_dataset_classes",
]
