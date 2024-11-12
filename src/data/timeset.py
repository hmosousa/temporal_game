from typing import Literal

import datasets

from src.prompts import NO_CONTEXT_PROMPT

_RELATION_MAP = {
    "BEFORE": "<",
    "AFTER": ">",
    "COEX": "=",
}


def load_timeset(split: Literal["valid", "test"] = "test"):
    if split == "valid":
        split = "validation"

    dataset = datasets.load_dataset(
        "kimihiroh/timeset", "pairwise", trust_remote_code=True, split=split
    )

    def process_example(example):
        src_id = example["id_arg1"]
        source = example["arg1"]

        tgt_id = example["id_arg2"]
        target = example["arg2"]

        context = (
            example["context"][: source["start"]]
            + f"<{src_id}>{source['mention']}</{src_id}>"
        )
        context += (
            example["context"][source["end"] : target["start"]]
            + f"<{tgt_id}>{target['mention']}</{tgt_id}>"
        )
        context += example["context"][target["end"] :]

        source_text = f"start <{src_id}>{source['mention']}</{src_id}>"
        target_text = f"start <{tgt_id}>{target['mention']}</{tgt_id}>"

        prompt = NO_CONTEXT_PROMPT.format(
            context=context, source=source_text, target=target_text
        )
        return {"text": prompt, "label": _RELATION_MAP[example["relation"]]}

    processed_dataset = dataset.map(process_example)
    processed_dataset = processed_dataset.remove_columns(dataset.column_names)
    return processed_dataset
