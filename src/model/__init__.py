from typing import Any, Dict

import torch
from transformers import AutoTokenizer

from src.model.classifier import Classifier
from src.constants import MODELS_DIR

MODELS = {
    "classifier": Classifier,
}


def load_model(
    model_name: str,
    config: Dict[str, Any],
    load_weights: bool = False,
    weights_filename: str = None,
):
    if model_name not in MODELS:
        raise ValueError(
            f"Model type '{model_name}' not found. Valid models are: {', '.join(MODELS.keys())}"
        )
    model = MODELS[model_name](**config)
    if load_weights:
        model_path = MODELS_DIR / config["output_path"] / weights_filename
        model.load_state_dict(torch.load(model_path))

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    return model, tokenizer


__all__ = ["Classifier", "load_model"]
