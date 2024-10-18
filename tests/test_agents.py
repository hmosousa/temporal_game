import pytest

from src.agents import BeforeAgent, LMAgentNoContext
from src.base import Timeline, Relation
from src.env import State, EntityPair


@pytest.fixture
def state():
    return State(
        context="They've been <ei2026>given</ei2026> a mandate from Switzerland to <ei2027>expand</ei2027> their U.S. chocolate operations.",
        entity_pairs=[
            EntityPair(source="start ei2026", target="start ei2027"),
            EntityPair(source="start ei2026", target="end ei2027"),
            EntityPair(source="end ei2026", target="start ei2027"),
            EntityPair(source="end ei2026", target="end ei2027"),
        ],
        timeline=Timeline(
            relations=[
                Relation(source="start ei2026", target="start ei2027", type="<"),
                Relation(source="start ei2026", target="end ei2027", type="<"),
                Relation(source="end ei2026", target="start ei2027", type="<"),
                Relation(source="end ei2026", target="end ei2027", type="<"),
            ]
        ),
    )


class TestBeforeAgent:
    def test_act(self, state):
        agent = BeforeAgent()
        assert agent.act(state) == Relation(
            source="start ei2026", target="start ei2027", type="<"
        )


class TestLMAgentNoContext:
    def test_act(self, state):
        agent = LMAgentNoContext("HuggingFaceTB/SmolLM-135M-Instruct")
        action = agent.act(state)
        assert action.source == "start ei2026"
        assert action.target == "start ei2027"
