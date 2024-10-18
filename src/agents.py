import transformers

from src.base import Relation, _RELATIONS
from src.env import State
from src.constants import HF_TOKEN


class BeforeAgent:
    """Agent that always classifies the first entity pair as before."""

    def act(self, state: State) -> Relation:
        pair = state["entity_pairs"][0]
        relation = Relation(
            source=pair["source"],
            target=pair["target"],
            type="<",
        )
        return relation


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


class LMAgentNoContext:
    """Agent that uses a language model to classify the first entity pair as before."""

    def __init__(self, model_name: str):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, token=HF_TOKEN
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, token=HF_TOKEN
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
            "temperature": 0.0,
            "return_full_text": False,
            "do_sample": False,
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
