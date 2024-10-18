import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple, TypedDict

import datasets

from src.base import Relation, Timeline


class EntityPair(TypedDict):
    source: str
    target: str


class State(TypedDict):
    context: str
    entity_pairs: List[EntityPair]
    timeline: Timeline


class TemporalGame:
    def __init__(self):
        self._data = datasets.load_dataset(
            "hugosousa/SmallTimelines", "one", split="train"
        )
        self._doc = None
        self._context = None
        self._entity_pairs = None
        self._timeline = None

    def reset(self, id: int = None) -> Tuple[State, Dict[str, Any]]:
        if id is None:
            self._id = random.randint(0, len(self._data) - 1)
        else:
            self._id = id
        self._doc = self._data[self._id]
        self._doc_timeline = Timeline(
            [
                Relation(
                    source=r["source"],
                    target=r["target"],
                    type=r["relation"],
                )
                for r in self._doc["timeline"]
            ]
        )
        self._doc_timeline = self._doc_timeline.closure()

        entity_pairs = []
        entities = deepcopy(self._doc["entities"])
        while entities:
            source = entities.pop()
            for target in entities:
                for source_prefix in ["start", "end"]:
                    for target_prefix in ["start", "end"]:
                        entity_pairs.append(
                            EntityPair(
                                source=f"{source_prefix} {source}",
                                target=f"{target_prefix} {target}",
                            )
                        )

        self._context = self._doc["context"]
        self._entity_pairs = entity_pairs
        self._timeline = Timeline()

        state = State(
            context=self._context,
            entity_pairs=self._entity_pairs,
            timeline=self._timeline,
        )

        self._info = {
            "id": self._id,
        }

        return state, self._info

    def step(self, action: Relation) -> Tuple[State, float, bool, bool, Dict[str, Any]]:
        """_summary_

        Args:
            action (_type_): _description_
        """
        self._timeline.add(action)
        if not self._timeline.is_valid:
            reward = 0
            terminated = True
            truncated = False
            state = State(
                context=self._context,
                entity_pairs=self._entity_pairs,
                timeline=self._timeline,
            )
            return state, reward, terminated, truncated, self._info

        terminated = False
        truncated = False

        # remove the relation from the entity pairs
        for entity_pair in self._entity_pairs:
            if (
                entity_pair["source"] == action.source
                and entity_pair["target"] == action.target
            ):
                self._entity_pairs.remove(entity_pair)
                break
            elif (
                entity_pair["source"] == action.target
                and entity_pair["target"] == action.source
            ):
                self._entity_pairs.remove(entity_pair)
                break
        else:
            raise ValueError(f"Entity pair {action.source} {action.target} not found")

        if len(self._entity_pairs) == 0:
            terminated = True

        # for making a step forward, we get 1 point
        reward = 1

        # for each inferable relation, we get 1 point
        n_relations = len(self._timeline)
        self._timeline = self._timeline.closure()
        n_inferable_relations = len(self._timeline) - n_relations
        reward += n_inferable_relations

        # if the relation is in the original annotation, we get 10 point
        if action in self._doc_timeline:
            reward += 10

        state = State(
            context=self._context,
            entity_pairs=self._entity_pairs,
            timeline=self._timeline,
        )
        return state, reward, terminated, truncated, self._info
