import torch
from torch.utils.data import DataLoader
from typing import Tuple
import datasets

from src.constants import DEVICE
from src.base import RELATIONS2ID


class SupervisedFineTuner:
    def __init__(
        self,
        model,
        tokenizer,
        lr: float,
        n_epochs: int,
        batch_size: int,
    ):
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, train_data: datasets.Dataset, valid_data: datasets.Dataset):
        train_data = self.prepare_dataset(train_data)
        valid_data = self.prepare_dataset(valid_data)

        for epoch in range(self.n_epochs):
            train_loss, train_acc = self.train_epoch(train_data)
            val_loss, val_acc = self.eval_epoch(valid_data)

            print(
                f"Epoch {epoch+1}/{self.config.n_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
            )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        total_correct = 0
        n = 0
        for batch in dataloader:
            labels = batch.pop("label").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}

            self.optimizer.zero_grad()
            logits = self.model(**inputs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            n += logits.size(0)

        loss = total_loss / n
        acc = total_correct / n
        return loss, acc

    def eval_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                labels = batch.pop("label").to(DEVICE)
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}

                logits = self.model(**inputs)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()
                n += logits.size(0)

        loss = total_loss / n
        acc = total_correct / n
        return loss, acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def prepare_dataset(self, dataset):
        dataset = dataset.map(lambda x: {"label": RELATIONS2ID[x["label"]]})
        dataset = self.tokenize_dataset(dataset)
        return dataset

    def tokenize_dataset(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        dataset = dataset.batch(self.batch_size)

        tokenized_dataset = dataset.map(tokenize_function)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset.set_format("torch")
        return tokenized_dataset
