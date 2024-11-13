from datetime import datetime
from typing import Dict, List

import datasets
import torch
import transformers
import wandb
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.base import ID2RELATIONS, RELATIONS2ID
from src.constants import DEVICE, HF_USERNAME
from src.data import balance_dataset_classes

transformers.logging.set_verbosity_error()
datasets.disable_progress_bar()


def generate_id() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


class SupervisedFineTuner:
    def __init__(
        self,
        model,
        tokenizer,
        lr: float,
        n_epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        cpu: bool = False,
        project_name: str = "Temporal Game",
        balance_classes: bool = False,
        use_wandb: bool = False,
        patience: int = None,
        push_to_hub: bool = False,
        hf_dir: str = None,
        **kwargs,
    ):
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        self.accelerator = Accelerator(cpu=cpu)
        self.lr = lr
        self.n_epochs = n_epochs

        # If the batch size is too big we use gradient accumulation
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.n_examples = 0

        self.use_wandb = use_wandb
        self.patience = patience
        self.early_stopping_counter = 0

        self._push_to_hub = push_to_hub
        self.run_id = generate_id()
        self.hf_dir = f"{HF_USERNAME}/{hf_dir}_{self.run_id}"

        self._balance_classes = balance_classes
        self._best_val_loss = float("inf")

        if self.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                name=hf_dir,
                project=project_name,
                config={
                    "learning_rate": lr,
                    "epochs": n_epochs,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "patience": patience,
                    "balance_classes": balance_classes,
                },
            )
            wandb.watch(self.model)

    def train(self, train_data: datasets.Dataset, valid_data: datasets.Dataset):
        if self.accelerator.is_main_process:
            print("Preparing dataloaders")

        valid_dataloader = self.get_dataloader(
            valid_data, batch_size=2 * self.batch_size
        )
        train_dataloader = self.get_dataloader(
            train_data, shuffle=True, batch_size=self.batch_size
        )

        T_0 = int(len(train_dataloader) / self.gradient_accumulation_steps)
        T_mult = 1
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0, T_mult
        )

        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            train_dataloader,
            valid_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            train_dataloader,
            valid_dataloader,
        )

        self.train_loop(train_dataloader, valid_dataloader)

    def train_loop(self, train_dataloader: DataLoader, valid_dataloader: DataLoader):
        log_step = 1
        for _ in range(self.n_epochs):
            self.model.train()

            progress_bar = (
                tqdm(
                    enumerate(train_dataloader),
                    desc="Training",
                    total=len(train_dataloader),
                )
                if self.accelerator.is_main_process
                else enumerate(train_dataloader)
            )

            for step, batch in progress_bar:
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                loss = loss / self.gradient_accumulation_steps
                self.accelerator.backward(loss)
                if step % self.gradient_accumulation_steps == 0:
                    self.lr_scheduler.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                batch_loss = outputs.loss.item()
                batch_correct = (
                    (outputs.logits.argmax(dim=-1) == batch["labels"]).sum().item()
                )
                batch_size = outputs.logits.size(0)
                self.n_examples += batch_size

                if self.use_wandb and self.accelerator.is_main_process:
                    wandb.log(
                        {
                            "n_examples": self.n_examples,
                            "train_loss": batch_loss / batch_size,
                            "train_acc": batch_correct / batch_size,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                        }
                    )

                if self.n_examples > log_step * 10_000:
                    val_metrics = self.eval(valid_dataloader)
                    log_step += 1

                    if self.accelerator.is_main_process:
                        if val_metrics["loss"] < self._best_val_loss:
                            self._best_val_loss = val_metrics["loss"]
                            self.early_stopping_counter = 0
                            tags = ["best_valid_loss"]
                        else:
                            tags = []

                        if self._push_to_hub:
                            msg = f"Save with {self.n_examples} examples"
                            self.push_to_hub(msg, tags)
                    else:
                        self.early_stopping_counter += 1

                    if self.use_wandb and self.accelerator.is_main_process:
                        wandb.log(
                            {
                                "n_examples": self.n_examples,
                                "val_loss": val_metrics["loss"],
                                "val_acc": val_metrics["acc"],
                                "best_val_loss": self._best_val_loss,
                                "val_acc_per_class": val_metrics["acc_per_class"],
                            }
                        )

                    if self.patience and self.early_stopping_counter >= self.patience:
                        if self.accelerator.is_main_process:
                            print(
                                f"Early stopping triggered after {self.n_examples} examples."
                            )
                        return  # Stop training if patience is exceeded

            # Reduce maximum learning rate
            self.lr_scheduler.scheduler.base_lrs[0] *= 0.8

    def eval(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        n = 0
        correct_per_class = {id: 0 for id in ID2RELATIONS}
        total_per_class = {id: 0 for id in ID2RELATIONS}
        with torch.no_grad():
            progress_bar = (
                tqdm(dataloader, desc="Evaluating")
                if self.accelerator.is_main_process
                else dataloader
            )

            for batch in progress_bar:
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                pred = outputs.logits.argmax(dim=-1)
                true = batch["labels"]
                batch_correct = (pred == true).sum().item()
                batch_size = outputs.logits.size(0)

                total_loss += outputs.loss.item()
                total_correct += batch_correct
                n += batch_size

                for id in ID2RELATIONS:
                    id_idxs = true == id
                    pred_id = pred[id_idxs]
                    true_id = true[id_idxs]
                    if pred_id.size(0) > 0:
                        correct_per_class[id] += (pred_id == true_id).sum().item()
                        total_per_class[id] += pred_id.size(0)

        epoch_loss = total_loss / n
        epoch_acc = total_correct / n
        accuracy_per_class = {
            relation: correct_per_class[id] / total_per_class[id]
            if total_per_class[id] > 0
            else 0.0
            for id, relation in ID2RELATIONS.items()
        }

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "acc_per_class": accuracy_per_class,
        }

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

        if self.use_wandb and self.accelerator.is_main_process:
            artifact = wandb.Artifact("trained_model", type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def push_to_hub(self, commit_message: str = "Update model", tags: List[str] = None):
        self.model.push_to_hub(self.hf_dir, commit_message=commit_message, tags=tags)
        self.tokenizer.push_to_hub(
            self.hf_dir, commit_message=commit_message, tags=tags
        )

    def balance_classes(self, dataset: datasets.Dataset) -> datasets.Dataset:
        return balance_dataset_classes(dataset, "labels")

    def get_dataloader(
        self, dataset: datasets.Dataset, batch_size: float, shuffle: bool = False
    ):
        # Rename the 'label' column to 'labels' which is the expected name for labels by the models of the
        # transformers library
        dataset = dataset.map(lambda x: {"label": RELATIONS2ID[x["label"]]})
        dataset = dataset.rename_column("label", "labels")

        if self._balance_classes:
            dataset = self.balance_classes(dataset)

        def tokenize_function(examples):
            outputs = self.tokenizer(examples["text"], truncation=True, max_length=None)
            return outputs

        # Apply the method we just defined to all the examples in all the splits of the dataset
        # starting with the main process first:
        with self.accelerator.main_process_first():
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
            )

        def collate_fn(examples):
            # When using mixed precision we want round multiples of 8/16
            if self.accelerator.mixed_precision == "fp8":
                pad_to_multiple_of = 16
            elif self.accelerator.mixed_precision != "no":
                pad_to_multiple_of = 8
            else:
                pad_to_multiple_of = None

            return self.tokenizer.pad(
                examples,
                padding="longest",
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            )

        # Instantiate dataloaders.
        dataloader = DataLoader(
            tokenized_dataset,
            shuffle=shuffle,
            collate_fn=collate_fn,
            batch_size=batch_size,
            drop_last=True,
        )

        return dataloader
