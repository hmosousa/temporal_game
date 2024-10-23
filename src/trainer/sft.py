import torch
from torch.utils.data import DataLoader
from typing import Tuple
import datasets
import wandb
from tqdm import tqdm

from src.constants import DEVICE, MODELS_DIR
from src.base import RELATIONS2ID


class SupervisedFineTuner:
    def __init__(
        self,
        model,
        tokenizer,
        lr: float,
        n_epochs: int,
        batch_size: int,
        project_name: str = "Temporal Game",
    ):
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.global_step = 0

        wandb.init(
            project=project_name,
            config={
                "learning_rate": lr,
                "epochs": n_epochs,
                "batch_size": batch_size,
            },
        )
        wandb.watch(self.model)

    def train(self, train_data: datasets.Dataset, valid_data: datasets.Dataset):
        train_data = self.prepare_dataset(train_data)
        valid_data = self.prepare_dataset(valid_data)

        best_val_loss = float("inf")
        for epoch in range(self.n_epochs):
            train_loss, train_acc = self.train_epoch(train_data)
            val_loss, val_acc = self.eval_epoch(valid_data)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = MODELS_DIR / self.model.name / f"{wandb.run.name}.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                self.save_model(model_path)

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

            print(
                f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
            )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        total_correct = 0
        n = 0
        for batch in tqdm(dataloader, desc="Training"):
            labels = batch.pop("label").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}

            self.optimizer.zero_grad()
            logits = self.model(**inputs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            batch_correct = (logits.argmax(dim=-1) == labels).sum().item()
            batch_size = logits.size(0)

            total_loss += batch_loss
            total_correct += batch_correct
            n += batch_size

            self.global_step += 1
            wandb.log(
                {
                    "step": self.global_step,
                    "step_train_loss": batch_loss / batch_size,
                    "step_train_acc": batch_correct / batch_size,
                }
            )

        epoch_loss = total_loss / n
        epoch_acc = total_correct / n
        return epoch_loss, epoch_acc

    def eval_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        n = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                labels = batch.pop("label").to(DEVICE)
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}

                logits = self.model(**inputs)
                loss = self.criterion(logits, labels)

                batch_loss = loss.item()
                batch_correct = (logits.argmax(dim=-1) == labels).sum().item()
                batch_size = logits.size(0)

                total_loss += batch_loss
                total_correct += batch_correct
                n += batch_size

        epoch_loss = total_loss / n
        epoch_acc = total_correct / n
        return epoch_loss, epoch_acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

        artifact = wandb.Artifact("trained_model", type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

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
