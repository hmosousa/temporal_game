from src.model import Classifier, load_model


def test_load_classifier():
    config = {"model_name": "google-bert/bert-base-uncased"}
    model, tokenizer = load_model("classifier", config)
    assert isinstance(model, Classifier)
    assert tokenizer is not None
