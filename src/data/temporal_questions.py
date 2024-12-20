from typing import Literal

import datasets


def load_temporal_questions(
    split: Literal["train", "valid", "test"],
) -> datasets.Dataset:
    """Used to train classification models."""
    dataset = datasets.load_dataset("hugosousa/TemporalQuestions", split=split)
    return dataset
