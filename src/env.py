import random
from typing import Any, Dict, List, Tuple, TypedDict

import datasets

from src.base import Relation, RELATIONS, Timeline


class EntityPair(TypedDict):
    source: str
    target: str


class State(TypedDict):
    context: str
    entity_pairs: List[EntityPair]
    timeline: Timeline


class TemporalGame:
    REWARD_PER_STEP = 1
    REWARD_PER_ANNOTATED_RELATION = 10

    def __init__(self, test: bool = False):
        if test:
            self._data = datasets.load_dataset(
                "hugosousa/SmallTimelines", "one", split="test"
            )
        else:
            self._data = datasets.load_dataset(
                "hugosousa/SmallTimelines", "one", split="train"
            )
        self._doc = None
        self._context = None
        self._entity_pairs = None
        self._timeline = None

    @property
    def num_docs(self) -> int:
        return len(self._data)

    @property
    def relation_types(self) -> List[str]:
        return RELATIONS

    @property
    def started(self) -> bool:
        return self._timeline is not None

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

        entities = self._doc["entities"]
        entity_pairs = [
            EntityPair(source=f"{s_prefix} {source}", target=f"{t_prefix} {target}")
            for i, source in enumerate(entities)
            for target in entities[i + 1 :]
            for s_prefix in ["start", "end"]
            for t_prefix in ["start", "end"]
        ]

        self._context = self._doc["context"]
        self._entity_pairs = entity_pairs
        self._timeline = Timeline()

        state = State(
            context=self._context,
            entity_pairs=self._entity_pairs,
            timeline=self._timeline,
        )

        # Calculate max reward
        n_annot_relations = len(self._doc_timeline)
        n_relations = len(self._entity_pairs)
        max_reward = (
            self.REWARD_PER_ANNOTATED_RELATION * n_annot_relations
            + self.REWARD_PER_STEP * (n_relations - n_annot_relations)
        )

        self._info = {
            "id": self._id,
            "doc_timeline": self._doc_timeline,
            "max_reward": max_reward,
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

        self._entity_pairs = [
            ep
            for ep in self._entity_pairs
            if not (
                (ep["source"] == action.source and ep["target"] == action.target)
                or (ep["source"] == action.target and ep["target"] == action.source)
            )
        ]

        if len(self._entity_pairs) == 0:
            terminated = True

        # for making a step forward
        reward = self.REWARD_PER_STEP

        # for each inferable relation
        n_relations = len(self._timeline)
        self._timeline = self._timeline.closure()
        n_inferable_relations = len(self._timeline) - n_relations
        reward += self.REWARD_PER_STEP * n_inferable_relations

        # if the relation is in the original annotation
        reward += (
            self.REWARD_PER_ANNOTATED_RELATION if action in self._doc_timeline else 0
        )

        state = State(
            context=self._context,
            entity_pairs=self._entity_pairs,
            timeline=self._timeline,
        )
        return state, reward, terminated, truncated, self._info
