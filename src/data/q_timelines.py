from typing import Literal

import datasets

_PROMPT = """Context:
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


def load_qtimelines_dataset(split: Literal["train", "valid"]) -> datasets.Dataset:
    data = datasets.load_dataset("hugosousa/SmallTimelines", "one", split="train")

    cutoff = int(len(data) * 0.8)
    if split == "train":
        data = data.select(range(0, cutoff))
    else:  # valid
        data = data.select(range(cutoff, len(data)))

    def process_example(example):
        new_examples = []
        for rel in example["timeline"]:
            prompt = _PROMPT.format(
                context=example["context"], source=rel["source"], target=rel["target"]
            )
            new_examples.append({"text": prompt, "label": rel["relation"]})
        return new_examples

    new_data = []
    for doc in data:
        new_data.extend(process_example(doc))

    return datasets.Dataset.from_list(new_data)
