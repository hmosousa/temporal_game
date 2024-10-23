import json
from pathlib import Path
from typing import Dict, Any

import yaml

from src.constants import CACHE_DIR, RESULTS_DIR, CONFIGS_DIR


def _save_dict(results: Dict, opath: Path):
    opath.parent.mkdir(exist_ok=True)
    with open(opath, "w") as f:
        json.dump(results, f, indent=4)


def store_results(results: Dict, agent_name: str):
    opath = RESULTS_DIR / f"{agent_name}.json"
    _save_dict(results, opath)


def cache_results(results: Dict, agent_name: str, doc_id: int):
    opath = CACHE_DIR / f"{agent_name}" / f"{doc_id}.json"
    _save_dict(results, opath)


def load_config(config_name: str) -> Dict[str, Any]:
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
