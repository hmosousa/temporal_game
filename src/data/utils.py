import random
from collections import Counter

import datasets


def balance_dataset_classes(dataset: datasets.Dataset, column: str) -> datasets.Dataset:
    # Count the number of samples per class
    class_counts = Counter(dataset[column])

    if len(class_counts) <= 1:
        return dataset

    max_count = max(class_counts.values())

    # Create a new balanced dataset
    balanced_data = {key: [] for key in dataset.features.keys()}

    for ref_label in class_counts:
        # Get all samples of the current class
        class_samples = [
            i for i, label in enumerate(dataset["label"]) if label == ref_label
        ]

        # Oversample the class samples to match the max_count
        oversampled_indices = random.choices(class_samples, k=max_count)

        for idx in oversampled_indices:
            for key in dataset.features.keys():
                balanced_data[key].append(dataset[key][idx])

    # Create a new dataset from the balanced data
    balanced_dataset = datasets.Dataset.from_dict(balanced_data)
    return balanced_dataset
