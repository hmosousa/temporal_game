import transformers

from src.agents.base import Agent
from src.base import _RELATIONS, Relation
from src.constants import HF_TOKEN
from src.env import State
from src.prompts import NO_CONTEXT_PROMPT


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
        prompt = NO_CONTEXT_PROMPT.format(
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
