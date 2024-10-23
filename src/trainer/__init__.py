from typing import Dict, Any

from src.trainer.sft import SupervisedFineTuner


TRAINERS = {
    "sft": SupervisedFineTuner,
}


def load_trainer(trainer_name: str, model, tokenizer, config: Dict[str, Any]):
    try:
        return TRAINERS[trainer_name](model, tokenizer, **config)
    except KeyError:
        raise ValueError(
            f"Trainer type '{trainer_name}' not found. Valid trainers are: {', '.join(TRAINERS.keys())}"
        )


__all__ = ["SupervisedFineTuner", "load_trainer"]
