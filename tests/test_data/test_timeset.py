import pytest
from datasets import Dataset

from src.data import load_dataset, load_timeset


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_timeset_test():
    testset = load_timeset("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 4_044


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_timeset_valid():
    validset = load_timeset("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 2_046


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_dataset_timeset():
    testset = load_dataset("timeset", "test")
    validset = load_dataset("timeset", "valid")

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)

    assert set(testset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}

    assert len(testset) == 4_044
    assert len(validset) == 2_046
