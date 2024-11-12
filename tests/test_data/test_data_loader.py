import pytest
from datasets import Dataset

from src.data import load_dataset, load_qtimelines


def test_load_q_timelines():
    trainset = load_qtimelines("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 75698


def test_load_q_timelines_augment():
    trainset = load_qtimelines("train", augment=True)
    assert len(trainset) == 75698 * 2


def test_load_dataset_q_timelines():
    trainset, validset = load_dataset("q_timelines")

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)

    assert set(trainset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}

    assert len(trainset) == 75698
    assert len(validset) == 6848


def test_load_dataset_q_timelines_augment():
    trainset, validset = load_dataset("q_timelines", {"augment": True})

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)

    assert set(trainset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}

    assert len(trainset) == 75698 * 2
    assert len(validset) == 6848


def test_load_dataset_invalid():
    with pytest.raises(ValueError) as excinfo:
        load_dataset("invalid_dataset", {})

    assert "Dataset invalid_dataset not found" in str(excinfo.value)
