from src.agents.base import Agent
from src.agents.before import BeforeAgent
from src.agents.lm import LMAgentNoContext
from src.agents.mcts import MCTSAgent
from src.agents.random import RandomAgent
from src.agents.trained import TrainedAgent
from src.env import TemporalGame


AGENT_MAP = {
    "before": BeforeAgent,
    "lm": LMAgentNoContext,
    "trained": TrainedAgent,
    "mcts": MCTSAgent,
    "random": RandomAgent,
}


def load_agent(
    agent_name: str,
    model_name: str = None,
    env: TemporalGame = None,
    num_simulations: int = 100,
) -> Agent:
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
        case "mcts":
            return agent_class(env, num_simulations)
        case _:
            return agent_class()


__all__ = [
    "Agent",
    "BeforeAgent",
    "LMAgentNoContext",
    "TrainedAgent",
    "RandomAgent",
    "load_agent",
]
