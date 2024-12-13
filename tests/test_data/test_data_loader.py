import pytest

from src.data import load_dataset


def test_load_dataset_invalid():
    with pytest.raises(ValueError) as excinfo:
        load_dataset("invalid_dataset", {})

    assert "Dataset invalid_dataset not found" in str(excinfo.value)
