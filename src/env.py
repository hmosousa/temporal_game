import random
import datasets

from copy import deepcopy
from base import Relation


class TemporalGame:
    def __init__(self):
        self._data = datasets.load_dataset(
            "hugosousa/SmallTimelines", "one", split="train"
        )
        self._doc = None

    def reset(self):
        self._id = random.randint(0, len(self._data) - 1)
        self._doc = self._data[self._id]

        relations = []
        entities = deepcopy(self._doc["entities"])
        while entities:
            source = entities.pop()
            for target in entities:
                for source_prefix in ["start", "end"]:
                    for target_prefix in ["start", "end"]:
                        relations.append(
                            f"{source_prefix}_{source['name']}",
                            f"{target_prefix}_{target['name']}",
                        )
        
        state = {
            "context": self._doc["context"],
            "relations": relations,
            "timeline": [],
        }

        info = {
            "id": self._id,
        }

        return state, info

    def step(self, action: Relation):
        """_summary_

        Args:
            action (_type_): _description_
        """
        pass
