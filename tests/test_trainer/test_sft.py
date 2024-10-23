import pytest
import torch
from transformers import AutoTokenizer
import datasets

from src.trainer.sft import SupervisedFineTuner
from src.model.classifier import Classifier


class TestSupervisedFineTuner:
    @pytest.fixture
    def model(self):
        return Classifier(model_name="bert-base-uncased")

    @pytest.fixture
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    @pytest.fixture
    def dummy_data(self):
        input_ids = torch.randint(0, 1000, (100, 10))
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        labels = torch.randint(0, 2, (100,))

        dataset = datasets.Dataset.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "label": labels,
            }
        )

        dataset = dataset.with_format("torch")
        return dataset

    def test_supervised_fine_tuner_init(self, model, tokenizer):
        sft = SupervisedFineTuner(model, tokenizer, lr=1e-5, n_epochs=3, batch_size=16)

        assert isinstance(sft.model, torch.nn.Module)
        assert sft.tokenizer == tokenizer
        assert sft.lr == 1e-5
        assert sft.n_epochs == 3
        assert sft.batch_size == 16
        assert isinstance(sft.optimizer, torch.optim.AdamW)
        assert isinstance(sft.criterion, torch.nn.CrossEntropyLoss)

    def test_train_epoch(self, model, tokenizer, dummy_data):
        sft = SupervisedFineTuner(model, tokenizer, lr=1e-5, n_epochs=3, batch_size=16)
        dataloader = dummy_data.batch(16)

        loss, acc = sft.train_epoch(dataloader)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 1

    def test_eval_epoch(self, model, tokenizer, dummy_data):
        sft = SupervisedFineTuner(model, tokenizer, lr=1e-5, n_epochs=3, batch_size=16)
        dataloader = dummy_data.batch(16)

        loss, acc = sft.eval_epoch(dataloader)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 1

    @pytest.mark.skip(reason="Takes too long to run.")
    def test_save_model(self, model, tokenizer, tmp_path):
        sft = SupervisedFineTuner(model, tokenizer, lr=1e-5, n_epochs=3, batch_size=16)
        save_path = tmp_path / "model.pth"

        sft.save_model(save_path)

        assert save_path.exists()
        loaded_state_dict = torch.load(save_path, weights_only=True)
        assert isinstance(loaded_state_dict, dict)
        assert set(loaded_state_dict.keys()) == set(model.state_dict().keys())
