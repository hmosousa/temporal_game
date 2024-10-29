from typing import Literal

import datasets

from src.constants import CACHE_DIR

from src.prompts import NO_CONTEXT_PROMPT


def load_qtimelines(split: Literal["train", "valid", "test"]) -> datasets.Dataset:
    """Used to train classification models."""
    cache_path = CACHE_DIR / "data" / f"q_timelines_{split}"
    if cache_path.exists():
        return datasets.load_from_disk(cache_path)

    if split == "test":
        data = datasets.load_dataset("hugosousa/SmallTimelines", "one", split="test")
    else:
        data = datasets.load_dataset("hugosousa/SmallTimelines", "one", split="train")

        cutoff = int(len(data) * 0.8)
        if split == "train":
            data = data.select(range(0, cutoff))
        else:  # valid
            data = data.select(range(cutoff, len(data)))

    def process_example(example):
        new_examples = []
        for rel in example["timeline"]:
            prompt = NO_CONTEXT_PROMPT.format(
                context=example["context"], source=rel["source"], target=rel["target"]
            )
            new_examples.append({"text": prompt, "label": rel["relation"]})
        return new_examples

    new_data = []
    for doc in data:
        new_data.extend(process_example(doc))

    dataset = datasets.Dataset.from_list(new_data)
    dataset.save_to_disk(cache_path)

    return dataset
