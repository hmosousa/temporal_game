from transformers import AutoModel
import torch.nn as nn

from src.base import N_RELATIONS


class Classifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.encoder.config.hidden_size, N_RELATIONS)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_output)
        return logits

    @property
    def name(self):
        return "classifier"
