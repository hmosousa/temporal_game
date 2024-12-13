import pytest
from datasets import Dataset

from src.data import load_dataset, load_qtimelines


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_q_timelines():
    trainset = load_qtimelines("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 57_184


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_q_timelines_augment():
    trainset = load_qtimelines("train", augment=True)
    assert len(trainset) == 57_184 * 2


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_dataset_q_timelines():
    trainset = load_dataset("q_timelines", "train")
    validset = load_dataset("q_timelines", "valid")

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)

    assert set(trainset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}

    assert len(trainset) == 57_184
    assert len(validset) == 4_924


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_dataset_q_timelines_augment():
    trainset = load_dataset("q_timelines", "train", {"augment": True})
    validset = load_dataset("q_timelines", "valid", {"augment": True})

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)

    assert set(trainset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}

    assert len(trainset) == 57_184 * 2
    assert len(validset) == 4_924 * 2
