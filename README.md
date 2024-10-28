# Temporal Game

Train an agent to build a timeline of events.

## Setup

Create `.env` file with the following:

```
HF_TOKEN=<your-huggingface-token>
HF_USERNAME=<your-huggingface-username>
```

For users:

```sh
conda create -p ./.conda python=3.11
conda activate ./.conda
pip install -e .
```

For developers:

```sh
conda create -p ./.conda python=3.11
conda activate ./.conda
pip install poetry
poetry install
poetry run pre-commit install
```

The developer setup installs Poetry for dependency management, installs all project dependencies, and sets up pre-commit hooks to maintain code quality and consistency across the project.

## Training

```sh
accelerate config
accelerate launch train_hf.py
```

### Profile the code

```sh
python -m cProfile -o profile.prof main.py
snakeviz profile.prof
```

## Results

### Classifier

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Label</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="5">LLAMA 1B</td>
            <td>&lt;</td>
            <td>57.50</td>
            <td>98.61</td>
            <td>72.64</td>
            <td>2088</td>
        </tr>
        <tr>
            <td>=</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>143</td>
        </tr>
        <tr>
            <td>&gt;</td>
            <td>68.15</td>
            <td>6.21</td>
            <td>11.39</td>
            <td>1481</td>
        </tr>
        <tr>
            <td>_</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>4</td>
        </tr>
        <tr>
            <td><strong>Overall</strong></td>
            <td><strong>57.88</strong></td>
            <td><strong>57.88</strong></td>
            <td><strong>45.35</strong></td>
            <td><strong>3716</strong></td>
        </tr>
        <tr>
            <td rowspan="5">BERT</td>
            <td>&lt;</td>
            <td>56.19</td>
            <td>100.00</td>
            <td>71.95</td>
            <td>2088</td>
        </tr>
        <tr>
            <td>=</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>143</td>
        </tr>
        <tr>
            <td>&gt;</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>1481</td>
        </tr>
        <tr>
            <td>_</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>4</td>
        </tr>
        <tr>
            <td><strong>Overall</strong></td>
            <td><strong>56.19</strong></td>
            <td><strong>56.19</strong></td>
            <td><strong>40.43</strong></td>
            <td><strong>3716</strong></td>
        </tr>
    </tbody>
</table>

### Game

| Model                   | Accuracy | Precision | Recall | F1    | Step Count | Reward |
|-------------------------|----------|-----------|--------|-------|------------|--------|
| before                  | 51.63    | 54.96     | 51.63  | 53.21 | 57.26      | 228.79 |
| llama-3.1-8b-instruct   | 36.14    | 43.03     | 36.14  | 38.61 | 40.72      | 139.23 |


## Load Models from Hugging Face


### Classifier
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "hugosousa/classifier_llama_1b", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("hugosousa/classifier_llama_1b")

inputs = tokenizer(["Hello, world!"], return_tensors="pt")
outputs = model(**inputs)
```

or using the pipeline

```python
import torch
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="hugosousa/classifier_llama_1b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print(classifier(["Hello, world!"]))

```
