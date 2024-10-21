from typing import Dict
import json

from src.constants import RESULTS_DIR


def store_results(results: Dict, agent_name: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{agent_name}.json", "w") as f:
        json.dump(results, f, indent=4)
