import json
import logging
from typing import Literal

import torch
from fire import Fire
from sklearn.metrics import classification_report

from src.constants import RESULTS_DIR
from src.data import load_dataset
from transformers import pipeline

logging.basicConfig(level=logging.INFO)


def main(
    model_name: str,
    dataset_name: Literal["q_timelines", "timeset", "levels"],
    verbose: bool = False,
):
    """Evaluate a model with a given configuration.

    Args:
        model_name: The HuggingFace name of the model to evaluate.
        dataset_name: The name of the dataset to evaluate on.
    """

    logging.info(f"Loading model {model_name}")
    classifier = pipeline(
        "text-classification",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    logging.info(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")

    logging.info("Getting predictions")
    preds = classifier(dataset["text"], batch_size=32)
    preds = [p["label"] for p in preds]
    labels = dataset["label"]

    logging.info("Calculating metrics")
    results = classification_report(labels, preds, output_dict=True)
    if verbose:
        print(classification_report(labels, preds))

    logging.info("Saving results")
    model_id = model_name.split("/")[-1]
    outpath = RESULTS_DIR / "classifier" / dataset_name / f"{model_id}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    Fire(main)
