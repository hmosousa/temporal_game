import os
from pathlib import Path

import dotenv

import torch

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "results"
CACHE_DIR = ROOT_DIR / "cache"
CONFIGS_DIR = ROOT_DIR / "configs"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NEW_TOKENS = [
    "<start_source>",
    "</start_source>",
    "<end_source>",
    "</end_source>",
    "<start_target>",
    "</start_target>",
    "<end_target>",
    "</end_target>",
]
