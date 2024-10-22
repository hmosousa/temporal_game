import json
import pandas as pd

from src.constants import RESULTS_DIR


def main():
    data = {}
    for file in RESULTS_DIR.glob("**/*.json"):
        with open(file, "r") as f:
            data[file.stem] = json.load(f)

    # print as table
    data = pd.DataFrame(data).T
    data["accuracy"] = data["accuracy"] * 100
    data["precision"] = data["precision"] * 100
    data["recall"] = data["recall"] * 100
    data["f1"] = data["f1"] * 100
    print(data.round(2))


if __name__ == "__main__":
    main()
