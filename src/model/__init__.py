from typing import Any, Dict

from transformers import AutoTokenizer

from src.model.classifier import Classifier

MODELS = {
    "classifier": Classifier,
}


def load_model(model_name: str, config: Dict[str, Any]):
    if model_name not in MODELS:
        raise ValueError(
            f"Model type '{model_name}' not found. Valid models are: {', '.join(MODELS.keys())}"
        )
    model = MODELS[model_name](**config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    return model, tokenizer


__all__ = ["Classifier", "load_model"]
