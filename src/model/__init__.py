from typing import Any, Dict

from src.model.classifier import load_classifier

MODEL_LOADERS = {
    "classifier": load_classifier,
}


def load_model(
    model_name: str,
    config: Dict[str, Any],
):
    if model_name not in MODEL_LOADERS:
        raise ValueError(
            f"Model type '{model_name}' not found. Valid models are: {', '.join(MODEL_LOADERS.keys())}"
        )

    model, tokenizer = MODEL_LOADERS[model_name](**config)
    return model, tokenizer


__all__ = ["load_classifier", "load_model"]
