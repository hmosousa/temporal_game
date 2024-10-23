from enum import Enum

from src.agents.base import Agent
from src.agents.before import BeforeAgent
from src.agents.lm import LMAgentNoContext
from src.agents.trained import TrainedAgent


class AgentType(Enum):
    BEFORE = "before"
    LM = "lm"
    TRAINED = "trained"


AGENT_MAP = {
    AgentType.BEFORE: BeforeAgent,
    AgentType.LM: LMAgentNoContext,
    AgentType.TRAINED: TrainedAgent,
}


def load_agent(agent_name: str, model_name: str = None) -> Agent:
    try:
        agent_type = AgentType(agent_name.lower())
        agent_class = AGENT_MAP[agent_type]
        return agent_class(model_name) if agent_type == AgentType.LM else agent_class()
    except ValueError:
        raise ValueError(
            f"Agent '{agent_name}' not found. Valid agents are: {', '.join([a.value for a in AgentType])}"
        )


__all__ = [
    "Agent",
    "AgentType",
    "BeforeAgent",
    "LMAgentNoContext",
    "TrainedAgent",
    "load_agent",
]
