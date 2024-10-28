from typing import Any, Dict

import yaml

from src.constants import CONFIGS_DIR


def load_config(config_name: str) -> Dict[str, Any]:
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
