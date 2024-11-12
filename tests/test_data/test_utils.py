from collections import Counter

import datasets
import pytest

from src.data.utils import balance_dataset_classes, get_entity_mapping


@pytest.fixture
def imbalanced_dataset():
    # Create a simple imbalanced dataset
    data = {
        "text": ["sample1", "sample2", "sample3", "sample4", "sample5"],
        "label": [0, 0, 0, 1, 1],  # 3 samples of class 0, 2 samples of class 1
    }
    return datasets.Dataset.from_dict(data)


def test_balance_classes_basic(imbalanced_dataset):
    balanced = balance_dataset_classes(imbalanced_dataset, "label")

    # Count samples per class in balanced dataset
    label_counts = Counter(balanced["label"])

    # Check if the number of samples per class equals the maximum count from original dataset
    assert label_counts[0] == 3
    assert label_counts[1] == 3


def test_balance_classes_maintains_features(imbalanced_dataset):
    balanced = balance_dataset_classes(imbalanced_dataset, "label")

    # Check if all original features are preserved
    assert set(balanced.features.keys()) == set(imbalanced_dataset.features.keys())

    # Check if all samples have corresponding text values
    assert len(balanced["text"]) == len(balanced["label"])
    assert all(isinstance(text, str) for text in balanced["text"])


def test_balance_classes_empty_dataset():
    empty_data = {"text": [], "label": []}
    empty_dataset = datasets.Dataset.from_dict(empty_data)

    balanced = balance_dataset_classes(empty_dataset, "label")
    assert len(balanced) == 0


def test_balance_classes_single_class():
    single_class_data = {
        "text": ["sample1", "sample2", "sample3"],
        "label": [0, 0, 0],  # Only one class
    }
    dataset = datasets.Dataset.from_dict(single_class_data)

    balanced = balance_dataset_classes(dataset, "label")
    assert len(balanced) == len(dataset)
    assert all(label == 0 for label in balanced["label"])


def test_balance_classes_invalid_column(imbalanced_dataset):
    with pytest.raises(KeyError):
        balance_dataset_classes(imbalanced_dataset, "invalid_column")


def test_get_entity_mapping_simple():
    context = "The <e1>New York Times</e1> is a newspaper."
    mapping = get_entity_mapping(context)
    assert mapping == {"e1": "New York Times"}


def test_get_entity_mapping_multiple():
    context = "The <e1>New York Times</e1> is a newspaper. <ei2>The Wall Street Journal</ei2> is a newspaper. <t1>2024</t1> is a year."
    mapping = get_entity_mapping(context)
    assert mapping == {
        "e1": "New York Times",
        "ei2": "The Wall Street Journal",
        "t1": "2024",
    }
