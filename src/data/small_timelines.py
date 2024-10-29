from typing import Literal

import datasets

from src.constants import CACHE_DIR


def load_small_timelines(split: Literal["train", "test"]) -> datasets.Dataset:
    """Used to train classification models."""
    cache_path = CACHE_DIR / "data" / f"small_timelines_{split}"
    if cache_path.exists():
        return datasets.load_from_disk(cache_path)

    if split == "test":
        dataset = datasets.load_dataset("hugosousa/SmallTimelines", "one", split="test")
    else:
        dataset = datasets.load_dataset(
            "hugosousa/SmallTimelines", "one", split="train"
        )

    dataset.save_to_disk(cache_path)
    return dataset
