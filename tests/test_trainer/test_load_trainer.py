from src.model import load_model_train
from src.trainer import load_trainer, SupervisedFineTuner


def test_load_sft():
    model_config = {"model_name": "google-bert/bert-base-uncased"}
    model, tokenizer = load_model_train("classifier", model_config)

    trainer_config = {
        "lr": 1e-3,
        "n_epochs": 1,
        "batch_size": 1,
        "output_path": "test",
        "gradient_accumulation_steps": 1,
    }
    trainer = load_trainer("sft", model, tokenizer, trainer_config)
    assert isinstance(trainer, SupervisedFineTuner)
