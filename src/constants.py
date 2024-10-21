import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "results"
