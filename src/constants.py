import os
from pathlib import Path

import torch
import dotenv

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "results"
CACHE_DIR = ROOT_DIR / "cache"
CONFIGS_DIR = ROOT_DIR / "configs"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
