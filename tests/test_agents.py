import pytest

from src.agents import BeforeAgent, LMAgentNoContext, RandomAgent, TrainedAgent
from src.base import Relation, Timeline
from src.constants import HF_USERNAME
from src.env import EntityPair, State


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
    @pytest.mark.skip(reason="This test is slow and should be run manually")
    def test_act(self, state):
        agent = LMAgentNoContext("HuggingFaceTB/SmolLM-135M-Instruct")
        action = agent.act(state)
        assert action.source == "start ei2026"
        assert action.target == "start ei2027"


class TestTrainedAgent:
    def test_act(self, state):
        agent = TrainedAgent(f"{HF_USERNAME}/classifier_llama_1b")
        action = agent.act(state)
        assert action.source == "start ei2026"
        assert action.target == "start ei2027"


class TestRandomAgent:
    def test_act_42(self, state):
        agent = RandomAgent()
        action = agent.act(state, seed=42)
        assert action.source == "start ei2026"
        assert action.target == "start ei2027"
        assert action.type == "<"

    def test_act_43(self, state):
        agent = RandomAgent()
        action = agent.act(state, seed=43)
        assert action.source == "start ei2026"
        assert action.target == "start ei2027"
        assert action.type == "="
