from src.agents.base import Agent, AgentType, AGENT_MAP
from src.agents.before import BeforeAgent
from src.agents.lm import LMAgentNoContext


AGENT_MAP.update(
    {
        AgentType.BEFORE: BeforeAgent,
        AgentType.LM: LMAgentNoContext,
    }
)


def load_agent(agent_name: str, model_name: str = None) -> Agent:
    try:
        agent_type = AgentType(agent_name.lower())
        agent_class = AGENT_MAP[agent_type]
        return agent_class(model_name) if agent_type == AgentType.LM else agent_class()
    except ValueError:
        raise ValueError(
            f"Agent '{agent_name}' not found. Valid agents are: {', '.join([a.value for a in AgentType])}"
        )


__all__ = ["Agent", "AgentType", "BeforeAgent", "LMAgentNoContext", "load_agent"]
