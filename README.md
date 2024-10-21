# Temporal Game

Train an agent to build a timeline of events.

## Setup

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
