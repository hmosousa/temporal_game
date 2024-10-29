import random

from src.agents.base import Agent
from src.base import Relation, RELATIONS
from src.env import State


class RandomAgent(Agent):
    @property
    def name(self) -> str:
        return "random"

    def act(self, state: State, seed: int = None) -> Relation:
        random.seed(seed)
        entity_pair = random.choice(state["entity_pairs"])
        relation_type = random.choice(RELATIONS)
        return Relation(
            source=entity_pair["source"],
            target=entity_pair["target"],
            type=relation_type,
        )
