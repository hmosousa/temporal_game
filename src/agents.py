from typing import Dict, Type
from enum import Enum

import transformers

from src.base import _RELATIONS, Relation
from src.constants import HF_TOKEN
from src.env import State


class Agent:
    def act(self, state: State) -> Relation:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


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


_NO_CONTEXT_PROMPT = """Context:
{context}

Question:
What is the temporal relation between the {source} and the {target}?

Options:
<, in case the {source} happens before the {target}
>, in case the {source} happens after the {target}
=, in case the {source} happens the same time as the {target}
-, in case the {source} happens not related to the {target}

Answer:
"""


class LMAgentNoContext(Agent):
    """Agent that uses a language model to classify the first entity pair as before."""

    def __init__(self, model_name: str):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
            device_map="auto",
        )
        self._pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )

    def act(self, state: State) -> Relation:
        pair = state["entity_pairs"][0]
        prompt = _NO_CONTEXT_PROMPT.format(
            context=state["context"], source=pair["source"], target=pair["target"]
        )

        generation_args = {
            "max_new_tokens": 1,
            "return_full_text": False,
            "do_sample": False,
            "temperature": 0.0,
        }

        relation_type = self._pipe(prompt, **generation_args)[0]["generated_text"][-1]

        if relation_type not in _RELATIONS:
            relation_type = "-"

        relation = Relation(
            source=pair["source"],
            target=pair["target"],
            type=relation_type,
        )
        return relation

    @property
    def name(self) -> str:
        return "lm"


class AgentType(Enum):
    BEFORE = "before"
    LM = "lm"


AGENT_MAP: Dict[AgentType, Type[Agent]] = {
    AgentType.BEFORE: BeforeAgent,
    AgentType.LM: LMAgentNoContext,
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
