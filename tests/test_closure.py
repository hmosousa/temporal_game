from src.base import INVERT_RELATION
from src.closure import compute_temporal_closure


def _compare_results(result, expected):
    if len(result) != len(expected):
        return False

    for relation in result:
        inverted_relation = {
            "source": relation["target"],
            "target": relation["source"],
            "relation": INVERT_RELATION[relation["relation"]],
        }
        if relation not in expected and inverted_relation not in expected:
            return False
    return True


def test_simple_less_than():
    relations = [
        {"source": "A", "target": "B", "relation": "<"},
        {"source": "B", "target": "C", "relation": "<"},
    ]
    result = compute_temporal_closure(relations)
    expected = [
        {"source": "A", "target": "B", "relation": "<"},
        {"source": "B", "target": "C", "relation": "<"},
        {"source": "A", "target": "C", "relation": "<"},
    ]
    assert _compare_results(result, expected)


def test_greater_than():
    relations = [{"source": "A", "target": "B", "relation": ">"}]
    result = compute_temporal_closure(relations)
    expected = [{"source": "B", "target": "A", "relation": "<"}]
    assert _compare_results(result, expected)


def test_equal_to():
    relations = [
        {"source": "A", "target": "B", "relation": "="},
        {"source": "B", "target": "C", "relation": "<"},
    ]
    result = compute_temporal_closure(relations)
    expected = [
        {"source": "A", "target": "B", "relation": "="},
        {"source": "B", "target": "C", "relation": "<"},
        {"source": "A", "target": "C", "relation": "<"},
    ]
    assert _compare_results(result, expected)


def test_null_relation():
    relations = [
        {"source": "A", "target": "B", "relation": "-"},
        {"source": "B", "target": "C", "relation": "<"},
    ]
    result = compute_temporal_closure(relations)
    expected = [
        {"source": "A", "target": "B", "relation": "-"},
        {"source": "B", "target": "C", "relation": "<"},
        {"source": "A", "target": "C", "relation": "-"},
    ]
    assert _compare_results(result, expected)


def test_multi_hop():
    relations = [
        {"source": "A", "target": "B", "relation": "<"},
        {"source": "B", "target": "C", "relation": "<"},
        {"source": "C", "target": "D", "relation": "<"},
    ]
    result = compute_temporal_closure(relations)
    expected = [
        {"source": "A", "target": "B", "relation": "<"},
        {"source": "B", "target": "C", "relation": "<"},
        {"source": "C", "target": "D", "relation": "<"},
        {"source": "A", "target": "C", "relation": "<"},
        {"source": "A", "target": "D", "relation": "<"},
        {"source": "B", "target": "D", "relation": "<"},
    ]
    assert _compare_results(result, expected)


def test_complex_scenario():
    relations = [
        {"source": "A", "target": "B", "relation": "<"},
        {"source": "B", "target": "C", "relation": "="},
        {"source": "C", "target": "D", "relation": "<"},
    ]
    result = compute_temporal_closure(relations)
    expected = [
        {"source": "A", "target": "B", "relation": "<"},
        {"source": "A", "target": "C", "relation": "<"},
        {"source": "A", "target": "D", "relation": "<"},
        {"source": "B", "target": "C", "relation": "="},
        {"source": "B", "target": "D", "relation": "<"},
        {"source": "C", "target": "D", "relation": "<"},
    ]
    assert _compare_results(result, expected)
