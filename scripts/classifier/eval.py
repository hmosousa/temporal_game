import json

import torch
from fire import Fire
from omegaconf import OmegaConf
from sklearn.metrics import classification_report

from src.constants import CONFIGS_DIR, HF_USERNAME, RESULTS_DIR
from src.data import load_qtimelines, load_timeset
from transformers import pipeline


def main(config_path: str = "classifier/bert.yaml", verbose: bool = False):
    """Evaluate a model with a given configuration.

    Args:
        config_path: The path to the configuration file used to train the model.
    """
    config = OmegaConf.load(CONFIGS_DIR / config_path)
    hf_dir = f"{HF_USERNAME}/{config.trainer.params.hf_dir}"

    try:
        classifier = pipeline(
            "text-classification",
            model=hf_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except ValueError:
        classifier = pipeline(
            "text-classification",
            model=hf_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    q_timelines = load_qtimelines(split="test")
    timeset = load_timeset(split="test")
    for dataset in [q_timelines, timeset]:
        preds = classifier(dataset["text"], batch_size=32)
        preds = [p["label"] for p in preds]
        labels = dataset["label"]

        results = classification_report(labels, preds, output_dict=True)
        if verbose:
            print(classification_report(labels, preds))

        outpath = RESULTS_DIR / "classifier" / f"{config.trainer.params.hf_dir}.json"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    Fire(main)
