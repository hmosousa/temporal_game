import json

import pandas as pd

from src.constants import RESULTS_DIR


def main(store: bool = True):
    data = {}
    ipath = RESULTS_DIR / "agents"
    for file in ipath.glob("**/*.json"):
        if file.name == "results.json":
            continue
        with open(file, "r") as f:
            data[file.stem] = json.load(f)

    # print as table
    data = pd.DataFrame(data).T
    data["accuracy"] = data["accuracy"] * 100
    data["precision"] = data["precision"] * 100
    data["recall"] = data["recall"] * 100
    data["f1"] = data["f1"] * 100
    print(data.round(2))

    if store:
        opath = ipath / "results.json"
        with open(opath, "w") as f:
            json.dump(data.round(2).to_dict(), f, indent=4)


if __name__ == "__main__":
    main()
