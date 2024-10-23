from src.agents.base import Agent
from src.base import ID2RELATION, Relation
from src.env import State
from src.model import load_model
from src.utils import load_config
from src.constants import DEVICE
from src.prompts import _NO_CONTEXT_PROMPT


class TrainedAgent(Agent):
    """Agent that was trained the temporal relations independently."""

    def __init__(self):
        config = load_config("classifier")
        self.model, self.tokenizer = load_model(
            "classifier",
            config["model"]["params"],
            load_weights=True,
            weights_filename="misty-voice-22.pt",
        )
        self.model.to(DEVICE)

    def act(self, state: State) -> Relation:
        pair = state["entity_pairs"][0]
        prompt = _NO_CONTEXT_PROMPT.format(
            context=state["context"], source=pair["source"], target=pair["target"]
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        relation_id = outputs.argmax(dim=-1)[0].item()
        relation_type = ID2RELATION[relation_id]

        relation = Relation(
            source=pair["source"],
            target=pair["target"],
            type=relation_type,
        )
        return relation

    @property
    def name(self) -> str:
        return "trained"
