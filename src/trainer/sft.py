from typing import Tuple

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
import transformers

import wandb
from src.base import RELATIONS2ID
from src.constants import DEVICE, MODELS_DIR, HF_USERNAME

transformers.logging.set_verbosity_error()
datasets.disable_progress_bar()


class SupervisedFineTuner:
    def __init__(
        self,
        model,
        tokenizer,
        lr: float,
        n_epochs: int,
        batch_size: int,
        output_path: str,
        cpu: bool = False,
        project_name: str = "Temporal Game",
        use_wandb: bool = False,
        patience: int = 5,
        push_to_hub: bool = False,
        hf_dir: str = None,
        max_gpu_batch_size: int = None,
        **kwargs,
    ):
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        self.accelerator = Accelerator(cpu=cpu)
        self.lr = lr
        self.n_epochs = n_epochs

        # If the batch size is too big we use gradient accumulation
        self.batch_size = batch_size
        self.gradient_accumulation_steps = 1
        if (
            max_gpu_batch_size is not None
            and batch_size > max_gpu_batch_size
            and self.accelerator.distributed_type != DistributedType.XLA
        ):
            self.gradient_accumulation_steps = self.batch_size // max_gpu_batch_size
            self.batch_size = max_gpu_batch_size

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.global_step = 0
        self.output_path = output_path

        self.use_wandb = use_wandb
        self.patience = patience
        self.early_stopping_counter = 0

        self._push_to_hub = push_to_hub
        self.hf_dir = f"{HF_USERNAME}/{hf_dir}"

        if self.use_wandb and self.accelerator.is_main_process:
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
        if self.accelerator.is_main_process:
            print("Preparing dataloaders")

        valid_dataloader = self.get_dataloader(
            valid_data, batch_size=2 * self.batch_size
        )
        train_dataloader = self.get_dataloader(
            train_data, shuffle=True, batch_size=self.batch_size
        )

        self.model, self.optimizer, train_dataloader, valid_dataloader = (
            self.accelerator.prepare(
                self.model, self.optimizer, train_dataloader, valid_dataloader
            )
        )

        best_val_loss = float("inf")
        for epoch in range(self.n_epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.eval_epoch(valid_dataloader)

            if val_loss < best_val_loss and self.accelerator.is_main_process:
                best_val_loss = val_loss
                self.early_stopping_counter = 0
                model_path = MODELS_DIR / self.output_path
                model_path.parent.mkdir(parents=True, exist_ok=True)
                self.save_model(model_path)
                if self._push_to_hub:
                    self.push_to_hub()
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.patience:
                if self.accelerator.is_main_process:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                break  # Stop training if patience is exceeded

            if self.use_wandb and self.accelerator.is_main_process:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }
                )

            if self.accelerator.is_main_process:
                print(
                    f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        total_correct = 0
        n = 0
        total_steps = len(dataloader)

        progress_bar = (
            tqdm(enumerate(dataloader), desc="Training", total=total_steps)
            if self.accelerator.is_main_process
            else enumerate(dataloader)
        )

        for step, batch in progress_bar:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.loss
            loss = loss / self.gradient_accumulation_steps
            self.accelerator.backward(loss)
            if step % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            batch_loss = loss.item()
            batch_correct = (
                (outputs.logits.argmax(dim=-1) == batch["labels"]).sum().item()
            )
            batch_size = outputs.logits.size(0)

            total_loss += batch_loss
            total_correct += batch_correct
            n += batch_size

            self.global_step += 1
            if self.use_wandb and self.accelerator.is_main_process:
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
            progress_bar = (
                tqdm(dataloader, desc="Evaluating")
                if self.accelerator.is_main_process
                else dataloader
            )

            for batch in progress_bar:
                batch.to(self.accelerator.device)

                outputs = self.model(**batch)

                batch_correct = (
                    (outputs.logits.argmax(dim=-1) == batch["labels"]).sum().item()
                )
                batch_size = outputs.logits.size(0)

                total_loss += outputs.loss.item()
                total_correct += batch_correct
                n += batch_size

        epoch_loss = total_loss / n
        epoch_acc = total_correct / n
        return epoch_loss, epoch_acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

        if self.use_wandb and self.accelerator.is_main_process:
            artifact = wandb.Artifact("trained_model", type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def push_to_hub(self):
        self.model.push_to_hub(self.hf_dir)
        self.tokenizer.push_to_hub(self.hf_dir)

    def get_dataloader(
        self, dataset: datasets.Dataset, batch_size: float, shuffle: bool = False
    ):
        # Rename the 'label' column to 'labels' which is the expected name for labels by the models of the
        # transformers library
        dataset = dataset.map(lambda x: {"label": RELATIONS2ID[x["label"]]})
        dataset = dataset.rename_column("label", "labels")

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
