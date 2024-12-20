import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import evaluate
import numpy as np
import transformers
from datasets import load_dataset

from src.constants import HF_TOKEN
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    EvalPrediction,
    set_seed,
    Trainer,
    TrainingArguments,
)
from transformers.utils import send_example_telemetry

DEBUG = False

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be used to shuffle the train dataset."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(
        default=None, metadata={"help": "The metric to use for evaluation."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [
            label for sample in raw_dataset[split]["label"] for label in sample
        ]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main(
    batch_size: int = 64,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 30,
):
    model_args = ModelArguments(
        model_name_or_path="HuggingFaceTB/SmolLM-135M",
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    data_args = DataTrainingArguments(
        dataset_name="hugosousa/QTimelines",
        dataset_config_name="default",
        max_seq_length=2048,  # TODO: Get from model
        pad_to_max_length=False,  # To pad each batch at runtime
        shuffle_train_dataset=True,
        shuffle_seed=42,
        max_train_samples=100 if DEBUG else None,
        max_eval_samples=100 if DEBUG else None,
        max_predict_samples=100 if DEBUG else None,
    )
    training_args = TrainingArguments(
        output_dir="models/SmolLM-135M-QTimelines",
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={
            "num_cycles": 1,
        },
        warmup_ratio=0.05,  # set here instead on the scheduler
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        logging_dir="logs",
        logging_steps=1,
        save_total_limit=2,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        bf16=True,
        push_to_hub=True,
        hub_model_id="hugosousa/SmolLM-135M-QTimelines",
        hub_strategy="every_save",
        hub_token=HF_TOKEN,
        do_train=True,
        do_eval=True,
        do_predict=False,
        report_to="wandb",
    )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    # if (
    #     os.path.isdir(training_args.output_dir)
    #     and training_args.do_train
    #     and not training_args.overwrite_output_dir
    # ):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif (
    #         last_checkpoint is not None and training_args.resume_from_checkpoint is None
    #     ):
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    # Print some info about the dataset
    logger.info(f"Dataset loaded: {raw_datasets}")
    logger.info(raw_datasets)

    label_list = ["<", ">", "=", "-"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        label2id=label2id,
        id2label=id2label,
    )

    logger.info("setting problem type to multi label classification")
    config.problem_type = "multi_label_classification"

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label2id)  # BCELoss requires float as target type
        for label in labels:
            ids[label2id[label]] = 1.0
        return ids

    def preprocess_function(examples):
        result = tokenizer(
            examples["text"],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )
        if label2id is not None and "label" in examples:
            result["label"] = [
                multi_labels_to_ids(labels) for labels in examples["label"]
            ]
        return result

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        raw_datasets["train"] = train_dataset

    if training_args.do_eval:
        eval_dataset = raw_datasets["valid"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        raw_datasets["valid"] = eval_dataset

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        # remove label column if it exists
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        raw_datasets["test"] = predict_dataset

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["valid"]
        predict_dataset = raw_datasets["test"]

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    metric = evaluate.load(
        "f1", config_name="multilabel", cache_dir=model_args.cache_dir
    )
    logger.info(
        "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
    )

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        preds = np.array(
            [np.where(p > 0, 1, 0) for p in preds]
        )  # convert logits to multi-hot encoding
        # Micro F1 is commonly used in multi-label classification
        result = metric.compute(
            predictions=preds, references=p.label_ids, average="micro"
        )

        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16 or training_args.bf16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
        if "label" in predict_dataset.features:
            predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        ).predictions

        # Convert logits to multi-hot encoding. We compare the logits to 0 instead of 0.5, because the sigmoid is not applied.
        # You can also pass `preprocess_logits_for_metrics=lambda logits, labels: nn.functional.sigmoid(logits)` to the Trainer
        # and set p > 0.5 below (less efficient in this case)
        predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])

        output_predict_file = os.path.join(
            training_args.output_dir, "predict_results.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    # recover from multi-hot encoding
                    item = [label_list[i] for i in range(len(item)) if item[i] == 1]
                    writer.write(f"{index}\t{item}\n")

        logger.info("Predict results saved at {}".format(output_predict_file))
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
