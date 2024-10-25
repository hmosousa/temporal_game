from src.model import load_model


def test_load_classifier():
    config = {"model_name": "google-bert/bert-base-uncased"}
    model, tokenizer = load_model("classifier", config)
    assert model is not None
    assert tokenizer is not None
