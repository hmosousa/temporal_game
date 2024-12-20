import itertools
from typing import Dict, List, Literal, Tuple

from tieval.links import TLink

from src.closure import compute_temporal_closure

RELATIONS = ["<", ">", "=", "-"]

N_RELATIONS = len(RELATIONS)

INVERT_RELATION = {
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

ID2RELATIONS = {v: k for k, v in RELATIONS2ID.items()}


class Relation:
    def __init__(self, source: str, target: str, type: Literal["<", ">", "=", "-"]):
        if type not in RELATIONS:
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
            and self.type == INVERT_RELATION[other.type]
        ):
            return True
        return False

    def __ne__(self, other: "Relation") -> bool:
        return not self == other

    def __invert__(self) -> "Relation":
        return Relation(
            source=self.target, target=self.source, type=INVERT_RELATION[self.type]
        )

    def __hash__(self) -> int:
        tmp = sorted([self.source, self.target])
        if tmp[0] == self.source:
            return hash(tuple([self.source, self.target, self.type]))
        else:
            return hash(tuple([self.target, self.source, INVERT_RELATION[self.type]]))

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
        }


class Timeline:
    """If on_endpoints is True, add implicit relations between the start and end of each entity."""

    def __init__(
        self,
        relations: List[Relation] = None,
        on_endpoints: bool = True,
        tlinks: List[TLink] = None,
    ):
        if tlinks is not None:
            relations = self.tlinks2relations(tlinks)

        elif relations is None:
            relations = set()

        self.relations = set(relations)
        self.entities = self._get_entities()
        self._on_endpoints = on_endpoints
        if on_endpoints:
            self.relations.update(self._expand_relations())
        self._relation_dict = self._build_relation_dict()
        self._closure_cache = None

    def __str__(self) -> str:
        return "\n".join([str(relation) for relation in self.relations])

    def __repr__(self) -> str:
        return f"Timeline({self.relations})"

    def __eq__(self, other: "Timeline") -> bool:
        return self.relations == other.relations

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

    def _expand_relations(self) -> List[Dict]:
        """Add implicit relations between the start and end of each entity.
        This is useful for computing the temporal closure."""
        relations = []
        unique_entities = set(ent.split(" ")[1] for ent in self.entities)
        for entity in unique_entities:
            relations.append(
                Relation(source=f"start {entity}", target=f"end {entity}", type="<")
            )
        return relations

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
                ],
                on_endpoints=self._on_endpoints,
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
        self.relations.add(relation)
        self.entities = self._get_entities()
        key = tuple(sorted([relation.source, relation.target]))
        if key not in self._relation_dict:
            self._relation_dict[key] = []
        self._relation_dict[key].append(relation)
        self._closure_cache = None  # Invalidate the cache

    def __and__(self, other: "Timeline") -> "Timeline":
        return Timeline(
            list(self.relations & other.relations),
            on_endpoints=self._on_endpoints,
        )

    def _build_relation_dict(self) -> Dict[Tuple[str, str], List[Relation]]:
        relation_dict = {}
        for relation in self.relations:
            key = tuple(sorted([relation.source, relation.target]))
            if key not in relation_dict:
                relation_dict[key] = []
            relation_dict[key].append(relation)
        return relation_dict

    def to_dict(self) -> Dict:
        return {
            "relations": [relation.to_dict() for relation in self.relations],
            "entities": self.entities,
        }

    @classmethod
    def from_relations(cls, relations: List[Dict]) -> "Timeline":
        return cls([Relation(**r) for r in relations])

    @staticmethod
    def tlinks2relations(tlinks):
        relations = []
        for tlink in tlinks:
            if tlink.source == tlink.target:
                continue

            pr = [r if r is not None else "-" for r in tlink.relation.point.relation]

            relations += [
                Relation(f"start {tlink.source}", f"start {tlink.target}", pr[0]),
                Relation(f"start {tlink.source}", f"end {tlink.target}", pr[1]),
                Relation(f"end {tlink.source}", f"start {tlink.target}", pr[2]),
                Relation(f"end {tlink.source}", f"end {tlink.target}", pr[3]),
                Relation(f"start {tlink.source}", f"end {tlink.source}", "<"),
                Relation(f"start {tlink.target}", f"end {tlink.target}", "<"),
            ]

        return set(relations)
