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
accelerate launch scripts/classifier/train.py
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
            <td rowspan="5">LLAMA 1B (Balanced)</td>
            <td>&lt;</td>
            <td>55.74</td>
            <td>58.38</td>
            <td>57.03</td>
            <td>2088</td>
        </tr>
        <tr>
            <td>=</td>
            <td>7.74</td>
            <td>80.42</td>
            <td>14.12</td>
            <td>143</td>
        </tr>
        <tr>
            <td>&gt;</td>
            <td>100.00</td>
            <td>0.07</td>
            <td>0.13</td>
            <td>1481</td>
        </tr>
        <tr>
            <td>-</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>4</td>
        </tr>
        <tr>
            <td><strong>Overall</strong></td>
            <td><strong>71.47</strong></td>
            <td><strong>35.93</strong></td>
            <td><strong>32.64</strong></td>
            <td><strong>3716</strong></td>
        </tr>
                <tr>
            <td rowspan="5">LLAMA 3B (Balanced)</td>
            <td>&lt;</td>
            <td>55.93</td>
            <td>74.71</td>
            <td>63.97</td>
            <td>2088</td>
        </tr>
        <tr>
            <td>=</td>
            <td>8.17</td>
            <td>32.87</td>
            <td>13.09</td>
            <td>143</td>
        </tr>
        <tr>
            <td>&gt;</td>
            <td>65.22</td>
            <td>3.04</td>
            <td>5.81</td>
            <td>1481</td>
        </tr>
        <tr>
            <td>-</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>0.00</td>
            <td>4</td>
        </tr>
        <tr>
            <td><strong>Overall</strong></td>
            <td><strong>57.74</strong></td>
            <td><strong>44.46</strong></td>
            <td><strong>38.76</strong></td>
            <td><strong>3716</strong></td>
        </tr>
    </tbody>
</table>

### Game

<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Step Count</th>
            <th>Reward</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2">Naive</td>
            <td>Random</td>
            <td>14.94</td>
            <td>24.26</td>
            <td>14.94</td>
            <td>17.57</td>
            <td>16.35</td>
            <td>54.66</td>
        </tr>
        <tr>
            <td>Before</td>
            <td>51.63</td>
            <td>54.96</td>
            <td>51.63</td>
            <td>53.21</td>
            <td>57.26</td>
            <td>228.79</td>
        </tr>
        <tr>
            <td rowspan="2">MCTS</td>
            <td>MCTS 100</td>
            <td>19.29</td>
            <td>26.85</td>
            <td>19.29</td>
            <td>21.69</td>
            <td>18.80</td>
            <td>64.33</td>
        </tr>
        <tr>
            <td>MCTS 200</td>
            <td>24.21</td>
            <td>31.09</td>
            <td>24.21</td>
            <td>26.26</td>
            <td>21.68</td>
            <td>75.39</td>
        </tr>
        <tr>
            <td>Prompt</td>
            <td>Llama 3.1 8B Instruct</td>
            <td>36.14</td>
            <td>43.03</td>
            <td>36.14</td>
            <td>38.61</td>
            <td>40.72</td>
            <td>139.23</td>
        </tr>
        <tr>
            <td rowspan="3">Classifier</td>
            <td>BERT</td>
            <td>51.63</td>
            <td>54.96</td>
            <td>51.63</td>
            <td>53.21</td>
            <td>57.26</td>
            <td>228.79</td>
        </tr>
        <tr>
            <td>Llama 1b</td>
            <td>43.87</td>
            <td>48.30</td>
            <td>43.87</td>
            <td>45.83</td>
            <td>45.87</td>
            <td>179.11</td>
        </tr>
        <tr>
            <td>Llama 1b (Balanced)</td>
            <td>33.25</td>
            <td>35.66</td>
            <td>33.25</td>
            <td>34.39</td>
            <td>56.73</td>
            <td>168.81</td>
        </tr>
    </tbody>
</table>

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
