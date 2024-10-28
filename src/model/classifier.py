from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch

from src.base import N_RELATIONS, ID2RELATION, RELATIONS2ID


def load_classifier(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=N_RELATIONS,
        torch_dtype=torch.bfloat16,
        id2label=ID2RELATION,
        label2id=RELATIONS2ID,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
