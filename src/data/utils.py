import random
from collections import Counter

import datasets


def balance_dataset_classes(dataset: datasets.Dataset, column: str) -> datasets.Dataset:
    # Count the number of samples per class
    class_counts = Counter(dataset[column])

    if len(class_counts) <= 1:
        return dataset

    max_count = max(class_counts.values())

    label_datasets = []
    for ref_label in class_counts:
        label_dataset = dataset.filter(lambda x: x[column] == ref_label)
        oversampled_indices = random.choices(range(len(label_dataset)), k=max_count)
        label_dataset = label_dataset.select(oversampled_indices)
        label_datasets.append(label_dataset)
    balanced_dataset = datasets.concatenate_datasets(label_datasets)

    return balanced_dataset
