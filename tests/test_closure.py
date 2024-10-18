from src.closure import compute_temporal_closure
from src.base import _INVERT_RELATION


def _compare_results(result, expected):
    result = sorted(result, key=lambda x: (x["source"], x["target"]))
    for relation in result:
        inverted_relation = {
            "source": relation["target"],
            "target": relation["source"],
            "relation": _INVERT_RELATION[relation["relation"]],
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
    ]
    assert _compare_results(result, expected)


# def test_complex_scenario():
#     relations = [
#         {"source": "A", "target": "B", "relation": "<"},
#         {"source": "B", "target": "C", "relation": "="},
#         {"source": "C", "target": "D", "relation": "<"},
#         {"source": "E", "target": "F", "relation": ">"},
#         {"source": "G", "target": "H", "relation": "-"},
#     ]
#     result = compute_temporal_closure(relations)
#     expected = [
#         {"source": "A", "target": "B", "relation": "<"},
#         {"source": "A", "target": "C", "relation": "<"},
#         {"source": "A", "target": "D", "relation": "<"},
#         {"source": "B", "target": "C", "relation": "="},
#         {"source": "B", "target": "D", "relation": "<"},
#         {"source": "C", "target": "D", "relation": "<"},
#         {"source": "F", "target": "E", "relation": "<"},
#         {"source": "G", "target": "H", "relation": "-"},
#     ]
#     assert result == expected
