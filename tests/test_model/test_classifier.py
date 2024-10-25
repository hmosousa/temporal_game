import torch

from src.base import N_RELATIONS
from src.model.classifier import load_classifier


class TestClassifier:
    def test_classifier_initialization(self):
        model, tokenizer = load_classifier(model_name="bert-base-uncased")
        assert isinstance(model, torch.nn.Module)
        assert tokenizer is not None

    def test_classifier_output_shape(self):
        model, tokenizer = load_classifier(model_name="bert-base-uncased")
        texts = ["Hello, world!", "This is a test."]
        batch_size = len(texts)
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        output = model(**inputs)
        assert output.logits.shape == (batch_size, N_RELATIONS)
