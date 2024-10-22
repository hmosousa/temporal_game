# Temporal Game

Train an agent to build a timeline of events.

## Setup

Create `.env` file with the following:

```
HF_TOKEN=<your-huggingface-token>
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


### Profile the code

```sh
python -m cProfile -o profile.prof main.py
snakeviz profile.prof
```

## Results

| Model                   | Accuracy | Precision | Recall | F1    | Step Count | Reward |
|-------------------------|----------|-----------|--------|-------|------------|--------|
| before                  | 51.63    | 54.96     | 51.63  | 53.21 | 57.26      | 228.79 |
| llama-3.1-8b-instruct   | 36.14    | 43.03     | 36.14  | 38.61 | 40.72      | 139.23 |
