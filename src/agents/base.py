from src.base import Relation
from src.env import State


class Agent:
    def act(self, state: State) -> Relation:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
