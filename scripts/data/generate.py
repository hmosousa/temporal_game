"""Generate Temporal Questions dataset from TemporalEval3 corpus."""

import copy
import logging
from collections import Counter
from pathlib import Path

import datasets
import fire
import tieval.datasets
from sklearn.model_selection import train_test_split

from src.base import Timeline
from src.constants import HF_TOKEN
from tieval.entities import Timex
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"


def add_tags(text: str, entities: list, dct: Timex = None) -> str:
    entities = sorted(list(entities), key=lambda x: x.offsets[0])

    context = ""
    if dct:
        context = f"Documents creation time: <{dct.id}>{dct.text}</{dct.id}>\n"

    e_prev = 0
    for entity in entities:
        s, e = entity.offsets
        context += text[e_prev:s]
        context += f"<{entity.id}>{entity.text}</{entity.id}>"
        e_prev = e
    context += text[e:]
    return context


def doc2questions(doc, split: str = "train"):
    if split == "train":
        tlinks = doc.temporal_closure
    else:
        tlinks = doc.tlinks
        for tlink in tlinks:
            tlink.source = tlink.source.id
            tlink.target = tlink.target.id

    entities_map = {ent.id: ent for ent in doc.entities + [doc.dct]}

    samples = []
    for tlink in tlinks:
        if (
            tlink.source not in entities_map
            or tlink.target not in entities_map
            or tlink.source == tlink.target
        ):
            continue

        source = entities_map[tlink.source]
        target = entities_map[tlink.target]

        has_dct = False
        if source.is_dct:
            entities = [target]
            has_dct = True
        elif target.is_dct:
            entities = [source]
            has_dct = True
        else:
            entities = [source, target]

        offsets = [idx for ent in entities for idx in ent.offsets]

        min_offset = min(offsets)
        max_offset = max(offsets)

        # Get the context that contains the entities
        # By merging the sentences that contain the entities
        context = []
        min_sent_offset = None
        for sent in doc.sentences:
            s_sent, e_sent = sent.offsets
            if (
                s_sent <= min_offset <= e_sent
                or min_offset <= s_sent <= e_sent <= max_offset
                or s_sent <= max_offset <= e_sent
            ):
                context.append(str(sent))
                if min_sent_offset is None or s_sent < min_sent_offset:
                    min_sent_offset = s_sent
        context = " ".join(context)

        # Update entity offsets of the current context
        for idx, ent in enumerate(entities):
            ent_ = copy.deepcopy(ent)
            s_ent, e_ent = ent.offsets
            ent_.offsets = [s_ent - min_sent_offset, e_ent - min_sent_offset]
            entities[idx] = ent_

        if has_dct:
            context = add_tags(context, entities, doc.dct)
        else:
            context = add_tags(context, entities)

        timeline = Timeline(tlinks=[tlink], on_endpoints=True).to_dict()

        samples.append(
            {
                "id": f"{doc.name}",
                "context": context,
                "source": source.id,
                "target": target.id,
                "timeline": timeline["relations"],
            }
        )
    return samples


def transform_corpus(documents, split: str = "train"):
    # Transform the documents into pair-wise contexts
    # Each tlink has its own context
    samples = []
    for doc in tqdm(documents):
        samples += doc2questions(doc, split)

    # Transform the contexts to have the special tokens
    examples = []
    for sample in samples:
        src_tags = f"<{sample['source']}>", f"</{sample['source']}>"
        tgt_tags = f"<{sample['target']}>", f"</{sample['target']}>"
        for relation in sample["timeline"]:
            # Skip classification of self-relations (ex: start A -> end A)
            src_id = relation["source"].split(" ")[1]
            tgt_id = relation["target"].split(" ")[1]
            if src_id == tgt_id:
                continue

            if relation["source"].startswith("start"):
                new_src_tags = "<start_source>", "</start_source>"
            else:
                new_src_tags = "<end_source>", "</end_source>"

            if relation["target"].startswith("start"):
                new_tgt_tags = "<start_target>", "</start_target>"
            else:
                new_tgt_tags = "<end_target>", "</end_target>"

            context = (
                sample["context"]
                .replace(src_tags[0], new_src_tags[0])
                .replace(src_tags[1], new_src_tags[1])
                .replace(tgt_tags[0], new_tgt_tags[0])
                .replace(tgt_tags[1], new_tgt_tags[1])
            )

            examples.append(
                {
                    "text": context,
                    "label": relation["type"],
                }
            )

    return examples


def validate_dataset(dataset: datasets.Dataset):
    """Check if the dataset is valid."""

    # Drop any relation that appears more than once
    # Most likely a mistake in the dataset
    text_counter = Counter(dataset["text"])
    duplicates = [text for text, count in text_counter.items() if count > 1]
    dataset = dataset.filter(lambda x: x["text"] not in duplicates)

    return dataset


def drop_long_texts(dataset: datasets.Dataset):
    """Drop texts that are longer than 512 words."""
    dataset = dataset.filter(lambda x: len(x["text"].split()) <= 512)
    return dataset


def main(dataset_name: str = "tempeval_3", n_valid_samples: int = 5_000):
    corpus = tieval.datasets.read(dataset_name)

    test_examples = transform_corpus(corpus.test, split="test")
    dev_examples = transform_corpus(corpus.train, split="train")

    # Stratified split into train and validation
    train_examples, valid_examples = train_test_split(
        dev_examples,
        test_size=n_valid_samples,
        random_state=42,
        stratify=[example["label"] for example in dev_examples],
        shuffle=True,
    )

    logging.info("Pushing to hub")
    trainset = datasets.Dataset.from_list(train_examples)
    validset = datasets.Dataset.from_list(valid_examples)
    testset = datasets.Dataset.from_list(test_examples)

    trainset = validate_dataset(trainset)
    validset = validate_dataset(validset)

    # TODO: This is just to iterate fast. Should be removed.
    trainset = drop_long_texts(trainset)

    trainset.push_to_hub("hugosousa/TemporalQuestions", split="train", token=HF_TOKEN)
    validset.push_to_hub("hugosousa/TemporalQuestions", split="valid", token=HF_TOKEN)
    testset.push_to_hub("hugosousa/TemporalQuestions", split="test", token=HF_TOKEN)


if __name__ == "__main__":
    fire.Fire(main)
