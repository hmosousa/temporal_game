import itertools
from typing import List, Literal, Tuple

from src.closure import compute_temporal_closure

_RELATIONS = ["<", ">", "=", "-"]


_INVERT_RELATION = {
    "<": ">",
    ">": "<",
    "=": "=",
    "-": "-",
}


class Relation:
    def __init__(self, source: str, target: str, type: Literal["<", ">", "=", "-"]):
        if type not in _RELATIONS:
            if type == "_":
                type = "-"
            else:
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
        relations_dict = [
            {
                "source": relation.source,
                "target": relation.target,
                "relation": relation.type,
            }
            for relation in self.relations
        ]
        inferred_relations = compute_temporal_closure(relations_dict)
        inferred_timeline = Timeline(
            [
                Relation(
                    source=relation["source"],
                    target=relation["target"],
                    type=relation["relation"],
                )
                for relation in inferred_relations
            ]
        )
        return inferred_timeline

    def __getitem__(self, key: Tuple[str, str]) -> List[Relation]:
        source, target = key
        return [
            relation
            for relation in self.relations
            if (relation.source == source and relation.target == target)
            or (relation.source == target and relation.target == source)
        ]

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
