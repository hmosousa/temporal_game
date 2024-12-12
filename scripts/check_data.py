"""Prompt a language model to check if the answer is correct.

Having problem to run the script in multiple GPUs? Try the following: `export VLLM_WORKER_MULTIPROC_METHOD=spawn`
"""

import datasets

from src.constants import HF_TOKEN
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DATASET_NAME = "hugosousa/QTimelines"
MODEL_NAME = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16"
# MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

NUMBER_GPUS = 4
MAX_MODEL_LEN = 8192


PROMPT_TEMPLATE = "Question: {text}\nAnswer: {label}\nIs the answer correct?\n"
SYSTEM_PROMPT = "You are an annotator whose job is to check if the answer to the question is correct. Reply with 'yes' or 'no'."


def main():
    dataset = datasets.load_dataset(DATASET_NAME, split="test")

    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(**example)},
        ]
        for example in dataset
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    llm = LLM(
        model=MODEL_NAME, tensor_parallel_size=NUMBER_GPUS, max_model_len=MAX_MODEL_LEN
    )
    sampling_params = SamplingParams(max_tokens=1)
    outputs = llm.generate(prompts, sampling_params)

    generated_text = [output.outputs[0].text.lower() for output in outputs]
    dataset = dataset.add_column("correct", generated_text)
    dataset.push_to_hub("hugosousa/QTimelines-annotated", split="test", token=HF_TOKEN)


if __name__ == "__main__":
    main()
