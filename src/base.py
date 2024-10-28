import itertools
from typing import List, Literal, Tuple, Dict

from src.closure import compute_temporal_closure

_RELATIONS = ["<", ">", "=", "-"]

N_RELATIONS = len(_RELATIONS)

_INVERT_RELATION = {
    "<": ">",
    ">": "<",
    "=": "=",
    "-": "-",
}

RELATIONS2ID = {
    ">": 0,
    "<": 1,
    "=": 2,
    "-": 3,
}

ID2RELATION = {v: k for k, v in RELATIONS2ID.items()}


class Relation:
    def __init__(self, source: str, target: str, type: Literal["<", ">", "=", "-"]):
        if type not in _RELATIONS:
            raise ValueError(f"Invalid relation type: {type}")
        self.source = source
        self.target = target
        self.type = type

    def __str__(self) -> str:
        return f"{self.source} {self.type} {self.target}"

    def __repr__(self) -> str:
        return f"Relation({self.source}, {self.target}, {self.type})"

    def __eq__(self, other: "Relation") -> bool:
        if (
            self.source == other.source
            and self.target == other.target
            and self.type == other.type
        ):
            return True
        elif (
            self.source == other.target
            and self.target == other.source
            and self.type == _INVERT_RELATION[other.type]
        ):
            return True
        return False

    def __ne__(self, other: "Relation") -> bool:
        return not self == other

    def __invert__(self) -> "Relation":
        return Relation(
            source=self.target, target=self.source, type=_INVERT_RELATION[self.type]
        )


class Timeline:
    def __init__(self, relations: List[Relation] = None):
        if relations is None:
            relations = []
        self.relations = relations
        self.entities = self._get_entities()
        self._relation_dict = self._build_relation_dict()
        self._closure_cache = None

    def __str__(self) -> str:
        return "\n".join([str(relation) for relation in self.relations])

    def __repr__(self) -> str:
        return f"Timeline({self.relations})"

    def __eq__(self, other: "Timeline") -> bool:
        if len(self.relations) != len(other.relations):
            return False
        for relation in self.relations:
            if relation not in other.relations:
                return False
        return True

    def __ne__(self, other: "Timeline") -> bool:
        return not self == other

    def __len__(self) -> int:
        return len(self.relations)

    def __contains__(self, relation: Relation) -> bool:
        return relation in self.relations

    def _get_entities(self) -> List[str]:
        entities = set()
        for relation in self.relations:
            entities.add(relation.source)
            entities.add(relation.target)
        return sorted(list(entities))

    def closure(self) -> "Timeline":
        if self._closure_cache is None:
            relations_dict = [
                {
                    "source": relation.source,
                    "target": relation.target,
                    "relation": relation.type,
                }
                for relation in self.relations
            ]
            inferred_relations = compute_temporal_closure(relations_dict)
            self._closure_cache = Timeline(
                [
                    Relation(
                        source=relation["source"],
                        target=relation["target"],
                        type=relation["relation"],
                    )
                    for relation in inferred_relations
                ]
            )
        return self._closure_cache

    def __getitem__(self, key: Tuple[str, str]) -> List[Relation]:
        sorted_key = tuple(sorted(key))
        relations = self._relation_dict.get(sorted_key, [])
        relations = [
            relation if relation.source == key[0] else ~relation
            for relation in relations
        ]
        return relations

    @property
    def is_valid(self) -> bool:
        """Check if the timeline is valid"""
        tc = self.closure()

        for source, target in tc.possible_relation_pairs:
            if len(tc[source, target]) > 1:
                return False
        return True

    @property
    def invalid_relations(self) -> List[Relation]:
        tc = self.closure()
        return [
            relation
            for source, target in tc.possible_relation_pairs
            for relation in tc[source, target]
            if len(tc[source, target]) > 1
        ]

    @property
    def possible_relation_pairs(self) -> List[Tuple[str, str]]:
        return list(itertools.combinations(self.entities, 2))

    def add(self, relation: Relation) -> None:
        self.relations.append(relation)
        self.entities = self._get_entities()
        key = tuple(sorted([relation.source, relation.target]))
        if key not in self._relation_dict:
            self._relation_dict[key] = []
        self._relation_dict[key].append(relation)
        self._closure_cache = None  # Invalidate the cache

    def __and__(self, other: "Timeline") -> "Timeline":
        """
        example:
        a = Timeline([Relation("A", "B", "<"), Relation("B", "C", "<")])
        b = Timeline([Relation("A", "B", "<"), Relation("B", "D", "<")])
        a & b = Timeline([Relation("A", "B", "<")])
        """
        return Timeline(
            [relation for relation in self.relations if relation in other.relations]
        )

    def _build_relation_dict(self) -> Dict[Tuple[str, str], List[Relation]]:
        relation_dict = {}
        for relation in self.relations:
            key = tuple(sorted([relation.source, relation.target]))
            if key not in relation_dict:
                relation_dict[key] = []
            relation_dict[key].append(relation)
        return relation_dict
