from src.agents.base import Agent
from src.agents.before import BeforeAgent
from src.agents.lm import LMAgentNoContext
from src.agents.trained import TrainedAgent


AGENT_MAP = {
    "before": BeforeAgent,
    "lm": LMAgentNoContext,
    "trained": TrainedAgent,
}


def load_agent(agent_name: str, model_name: str = None) -> Agent:
    if agent_name not in AGENT_MAP:
        raise ValueError(
            f"Agent '{agent_name}' not found. Valid agents are: {', '.join(AGENT_MAP.keys())}"
        )

    agent_class = AGENT_MAP[agent_name]
    match agent_name:
        case "lm":
            return agent_class(model_name)
        case "trained":
            return agent_class(model_name)
        case _:
            return agent_class()


__all__ = [
    "Agent",
    "BeforeAgent",
    "LMAgentNoContext",
    "TrainedAgent",
    "load_agent",
]
