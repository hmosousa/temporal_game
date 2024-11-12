import pytest
from datasets import Dataset

from src.data import load_dataset, load_qtimelines, load_timeset


def test_load_q_timelines():
    trainset = load_qtimelines("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 63_840


def test_load_q_timelines_augment():
    trainset = load_qtimelines("train", augment=True)
    assert len(trainset) == 63_840 * 2


def test_load_dataset_q_timelines():
    trainset = load_dataset("q_timelines", "train")
    validset = load_dataset("q_timelines", "valid")

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)

    assert set(trainset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}

    assert len(trainset) == 63_840
    assert len(validset) == 5_816


def test_load_dataset_q_timelines_augment():
    trainset = load_dataset("q_timelines", "train", {"augment": True})
    validset = load_dataset("q_timelines", "valid", {"augment": True})

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)

    assert set(trainset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}

    assert len(trainset) == 63_840 * 2
    assert len(validset) == 5_816 * 2


def test_load_dataset_invalid():
    with pytest.raises(ValueError) as excinfo:
        load_dataset("invalid_dataset", {})

    assert "Dataset invalid_dataset not found" in str(excinfo.value)


def test_load_timeset_test():
    testset = load_timeset("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 4_044


def test_load_timeset_valid():
    validset = load_timeset("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 2_046


def test_load_dataset_timeset():
    testset = load_dataset("timeset", "test")
    validset = load_dataset("timeset", "valid")

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)
