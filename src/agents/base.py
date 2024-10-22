from typing import Dict, Type
from enum import Enum

from src.base import Relation
from src.env import State


class Agent:
    def act(self, state: State) -> Relation:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class AgentType(Enum):
    BEFORE = "before"
    LM = "lm"


AGENT_MAP: Dict[AgentType, Type[Agent]] = {}
