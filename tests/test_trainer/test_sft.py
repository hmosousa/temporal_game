import pytest
import torch
import datasets

from src.trainer.sft import SupervisedFineTuner, MAX_GPU_BATCH_SIZE
from src.model import load_classifier


class TestSupervisedFineTuner:
    @pytest.fixture
    def dummy_data(self):
        text = ["hi"] * 100
        labels = ["<"] * 100

        dataset = datasets.Dataset.from_dict(
            {
                "text": text,
                "label": labels,
            }
        )

        dataset = dataset.with_format("torch")
        return dataset

    def test_supervised_fine_tuner_init(self):
        model, tokenizer = load_classifier(model_name="bert-base-uncased")

        sft = SupervisedFineTuner(
            model, tokenizer, lr=1e-5, n_epochs=3, batch_size=16, output_path="test"
        )

        assert isinstance(sft.model, torch.nn.Module)
        assert sft.tokenizer == tokenizer
        assert sft.lr == 1e-5
        assert sft.n_epochs == 3
        assert sft.batch_size == MAX_GPU_BATCH_SIZE
        assert isinstance(sft.optimizer, torch.optim.AdamW)
        assert isinstance(sft.criterion, torch.nn.CrossEntropyLoss)

    def test_train_epoch(self, dummy_data):
        model, tokenizer = load_classifier(model_name="bert-base-uncased")
        sft = SupervisedFineTuner(
            model, tokenizer, lr=1e-5, n_epochs=3, batch_size=16, output_path="test"
        )
        dataloader = sft.get_dataloader(dummy_data, 16)

        loss, acc = sft.train_epoch(dataloader)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 1

    def test_eval_epoch(self, dummy_data):
        model, tokenizer = load_classifier(model_name="bert-base-uncased")
        sft = SupervisedFineTuner(
            model, tokenizer, lr=1e-5, n_epochs=3, batch_size=16, output_path="test"
        )
        dataloader = sft.get_dataloader(dummy_data, 16)

        loss, acc = sft.eval_epoch(dataloader)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 1

    @pytest.mark.skip(reason="Takes too long to run.")
    def test_save_model(self, tmp_path):
        model, tokenizer = load_classifier(model_name="bert-base-uncased")
        sft = SupervisedFineTuner(
            model, tokenizer, lr=1e-5, n_epochs=3, batch_size=16, output_path="test"
        )
        save_path = tmp_path / "model.pth"

        sft.save_model(save_path)

        assert save_path.exists()
        loaded_state_dict = torch.load(save_path, weights_only=True)
        assert isinstance(loaded_state_dict, dict)
        assert set(loaded_state_dict.keys()) == set(model.state_dict().keys())
