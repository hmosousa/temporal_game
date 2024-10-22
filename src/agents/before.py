from src.agents.base import Agent
from src.base import Relation
from src.env import State


class BeforeAgent(Agent):
    """Agent that always classifies the first entity pair as before."""

    def act(self, state: State) -> Relation:
        pair = state["entity_pairs"][0]
        relation = Relation(
            source=pair["source"],
            target=pair["target"],
            type="<",
        )
        return relation

    @property
    def name(self) -> str:
        return "before"
