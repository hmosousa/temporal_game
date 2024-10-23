import torch

from src.base import N_RELATIONS
from src.model.classifier import Classifier


class TestClassifier:
    def test_classifier_initialization(self):
        classifier = Classifier(model_name="bert-base-uncased")
        assert isinstance(classifier, Classifier)
        assert isinstance(classifier.encoder, torch.nn.Module)

    def test_classifier_output_shape(self):
        classifier = Classifier(model_name="bert-base-uncased")
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        output = classifier(input_ids=input_ids, attention_mask=attention_mask)
        assert output.shape == (batch_size, N_RELATIONS)
