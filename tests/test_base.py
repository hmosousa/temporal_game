import pytest

from src.base import _INVERT_RELATION, Relation, Timeline


class TestRelation:
    def test_relation_equality(self):
        r1 = Relation(source="A", target="B", type="<")
        r2 = Relation(source="A", target="B", type="<")
        r3 = Relation(source="B", target="A", type=">")
        r4 = Relation(source="A", target="B", type=">")

        assert r1 == r2
        assert r1 == r3
        assert r1 != r4

    def test_relation_inversion(self):
        relations = [
            (
                Relation(source="A", target="B", type="<"),
                Relation(source="B", target="A", type=">"),
            ),
            (
                Relation(source="X", target="Y", type=">"),
                Relation(source="Y", target="X", type="<"),
            ),
            (
                Relation(source="P", target="Q", type="="),
                Relation(source="Q", target="P", type="="),
            ),
            (
                Relation(source="M", target="N", type="-"),
                Relation(source="N", target="M", type="-"),
            ),
        ]

        for original, expected in relations:
            inverted = ~original
            assert inverted == expected
            assert inverted.source == expected.source
            assert inverted.target == expected.target
            assert inverted.type == expected.type

    def test_relation_invert_relation_consistency(self):
        for rel_type, inverted_type in _INVERT_RELATION.items():
            r = Relation(source="A", target="B", type=rel_type)
            inverted = ~r
            assert inverted.type == inverted_type

    def test_relation_invalid_type(self):
        with pytest.raises(ValueError):
            Relation(source="A", target="B", type="invalid")

    def test_relation_source_target_swap(self):
        r1 = Relation(source="A", target="B", type="<")
        r2 = Relation(source="B", target="A", type=">")
        assert r1 == r2

        r3 = Relation(source="X", target="Y", type="=")
        r4 = Relation(source="Y", target="X", type="=")
        assert r3 == r4

        r5 = Relation(source="P", target="Q", type="-")
        r6 = Relation(source="Q", target="P", type="-")
        assert r5 == r6

    def test_relation_inequality(self):
        r1 = Relation(source="A", target="B", type="<")
        r2 = Relation(source="B", target="C", type="<")
        r3 = Relation(source="A", target="B", type="=")

        assert r1 != r2
        assert r1 != r3
        assert r2 != r3

    def test_in_list(self):
        r1 = Relation(source="A", target="B", type="<")
        r2 = Relation(source="B", target="C", type="<")
        r3 = Relation(source="A", target="B", type="=")

        relations = [r1, r2, r3]
        assert r1 in relations
        assert r2 in relations
        assert r3 in relations

    def test_hash(self):
        r1 = Relation(source="A", target="B", type="<")
        r2 = Relation(source="B", target="A", type=">")
        assert hash(r1) == hash(r2)


class TestTimeline:
    @pytest.fixture
    def relations(self):
        return [
            Relation(source="A", target="B", type="<"),
            Relation(source="B", target="C", type="<"),
            Relation(source="C", target="D", type="<"),
            Relation(source="E", target="F", type=">"),
            Relation(source="G", target="H", type="-"),
        ]

    @pytest.fixture
    def relations_closure(self):
        return [
            Relation(source="A", target="B", type="<"),
            Relation(source="B", target="C", type="<"),
            Relation(source="A", target="C", type="<"),
            Relation(source="C", target="D", type="<"),
            Relation(source="A", target="D", type="<"),
            Relation(source="B", target="D", type="<"),
            Relation(source="E", target="F", type=">"),
            Relation(source="G", target="H", type="-"),
        ]

    def test_timeline_equality(self):
        t1 = Timeline(relations=[Relation(source="A", target="B", type="<")])
        t2 = Timeline(relations=[Relation(source="A", target="B", type="<")])
        t3 = Timeline(relations=[Relation(source="B", target="A", type=">")])
        t4 = Timeline(relations=[Relation(source="A", target="B", type="=")])

        assert t1 == t2
        assert t1 == t3
        assert not (t1 == t4)

    def test_timeline_valid_closure(self, relations, relations_closure):
        t = Timeline(relations=relations)
        expected_tc = Timeline(relations=relations_closure)
        tc = t.closure()
        assert tc == expected_tc

    def test_invalid_relations(self):
        relations = [
            Relation(source="A", target="B", type="<"),
            Relation(source="B", target="C", type="<"),
            Relation(source="A", target="C", type=">"),
            Relation(source="D", target="E", type=">"),
        ]
        timeline = Timeline(relations=relations)
        assert not timeline.is_valid
        assert len(timeline.invalid_relations) == 6

    def test_entities(self, relations):
        t = Timeline(relations=relations)
        assert t.entities == ["A", "B", "C", "D", "E", "F", "G", "H"]

    def test_possible_relation_pairs(self):
        t = Timeline(
            relations=[
                Relation(source="A", target="B", type="<"),
                Relation(source="B", target="C", type="<"),
            ]
        )
        assert t.possible_relation_pairs == [
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
        ]

    def test_get_item(self):
        t = Timeline(relations=[Relation(source="A", target="B", type="<")])
        assert t["A", "B"] == [Relation(source="A", target="B", type="<")]
        assert t["B", "A"] == [Relation(source="B", target="A", type=">")]

    def test_get_item_source_target(self):
        t = Timeline(relations=[Relation(source="A", target="B", type="<")])
        rel = t["A", "B"][0]
        assert rel.source == "A"
        assert rel.target == "B"
        assert rel.type == "<"

    def test_get_item_source_target_swap(self):
        t = Timeline(relations=[Relation(source="A", target="B", type="<")])
        rel = t["B", "A"][0]
        assert rel.source == "B"
        assert rel.target == "A"
        assert rel.type == ">"

    def test_len(self, relations):
        t = Timeline(relations=relations)
        assert len(t) == 5

    def test_contains(self):
        t = Timeline(
            relations=[
                Relation(source="A", target="B", type="<"),
            ]
        )
        assert Relation(source="A", target="B", type="<") in t
        assert Relation(source="B", target="A", type=">") in t
        assert Relation(source="B", target="C", type="<") not in t

    def test_add_relation(self):
        t = Timeline()

        t.add(Relation(source="A", target="B", type="<"))
        assert t == Timeline(relations=[Relation(source="A", target="B", type="<")])

        t.add(Relation(source="B", target="C", type="<"))
        assert t == Timeline(
            relations=[
                Relation(source="A", target="B", type="<"),
                Relation(source="B", target="C", type="<"),
            ]
        )
