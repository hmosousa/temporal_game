import transformers

from src.base import Relation
from src.env import State


class BeforeAgent:
    """Agent that always classifies the first entity pair as before."""

    def act(self, state: State) -> Relation:
        pair = state["entity_pairs"][0]
        relation = Relation(
            source=pair["source"],
            target=pair["target"],
            type="<",
        )
        return relation


class LMAgentNoContext:
    """Agent that uses a language model to classify the first entity pair as before."""

    def __init__(self, model_name: str):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def act(self, state: State) -> Relation:
        pair = state["entity_pairs"][0]
        pair
