import torch
from transformers import pipeline

from src.agents.base import Agent
from src.base import Relation
from src.env import State
from src.prompts import NO_CONTEXT_PROMPT


class TrainedAgent(Agent):
    """Agent that was trained the temporal relations independently."""

    def __init__(self, model_name: str):
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def act(self, state: State) -> Relation:
        pair = state["entity_pairs"][0]
        prompt = NO_CONTEXT_PROMPT.format(
            context=state["context"], source=pair["source"], target=pair["target"]
        )

        prediction = self.classifier(prompt)
        relation_type = prediction[0]["label"]

        relation = Relation(
            source=pair["source"],
            target=pair["target"],
            type=relation_type,
        )
        return relation

    @property
    def name(self) -> str:
        return "trained"
