from typing import Literal

import datasets

from src.base import INVERT_RELATION
from src.constants import CACHE_DIR
from src.data.utils import get_entity_mapping
from src.prompts import NO_CONTEXT_PROMPT


def load_levels(
    split: Literal["train", "valid", "test"],
    level: Literal[0, 1, 2],
    augment: bool = False,
    use_cache: bool = False,
) -> datasets.Dataset:
    """Used to train classification models."""
    if split in ["test"]:  # No augmentation for validation and test
        augment = False

    cache_path = CACHE_DIR / "data" / f"levels_{split}_{level}_{augment}"
    if use_cache and cache_path.exists():
        return datasets.load_from_disk(cache_path)

    if split == "test":
        data = datasets.load_dataset(
            "hugosousa/TemporalGamesLevels", "clean", split="test"
        )
    else:
        data = datasets.load_dataset(
            "hugosousa/TemporalGamesLevels", "clean", split="train"
        )

        cutoff = int(len(data) * 0.9)
        if split == "train":
            data = data.select(range(0, cutoff))
        else:  # valid
            data = data.select(range(cutoff, len(data)))

    def process_example(example, augment: bool = False):
        new_examples = []
        eid2text = get_entity_mapping(example[f"level_{level}"])
        for rel in example["timeline"]:
            src_endpoint, src_id = rel["source"].split(" ")
            src_text = eid2text[src_id]
            source = f"{src_endpoint} <{src_id}>{src_text}</{src_id}>"

            tgt_endpoint, tgt_id = rel["target"].split(" ")
            tgt_text = eid2text[tgt_id]
            target = f"{tgt_endpoint} <{tgt_id}>{tgt_text}</{tgt_id}>"

            if src_id == tgt_id:
                # The dataset contains relations between start and end of the same entity
                continue

            prompt = NO_CONTEXT_PROMPT.format(
                context=example[f"level_{level}"], source=source, target=target
            )
            new_examples.append({"text": prompt, "label": rel["relation"]})

            if augment:
                inverted_rel = INVERT_RELATION[rel["relation"]]
                inverted_prompt = NO_CONTEXT_PROMPT.format(
                    context=example[f"level_{level}"],
                    source=target,
                    target=source,
                )
                new_examples.append({"text": inverted_prompt, "label": inverted_rel})
        return new_examples

    new_data = []
    for doc in data:
        new_data.extend(process_example(doc, augment))

    dataset = datasets.Dataset.from_list(new_data)
    dataset.save_to_disk(cache_path)

    return dataset