from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.base import N_RELATIONS


def load_classifier(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=N_RELATIONS
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
