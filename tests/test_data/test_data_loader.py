import pytest
from datasets import Dataset

from src.data import load_dataset


def test_load_q_timelines():
    trainset, validset = load_dataset("q_timelines", {})

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)

    assert set(trainset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}

    assert len(trainset) > 0
    assert len(validset) > 0


def test_load_dataset_invalid():
    with pytest.raises(ValueError) as excinfo:
        load_dataset("invalid_dataset", {})

    assert "Dataset invalid_dataset not found" in str(excinfo.value)
