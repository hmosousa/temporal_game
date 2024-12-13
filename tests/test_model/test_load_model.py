from src.model import load_model_train


def test_load_classifier():
    config = {"model_name": "google-bert/bert-base-uncased"}
    model, tokenizer = load_model_train("classifier", config)
    assert model is not None
    assert tokenizer is not None
