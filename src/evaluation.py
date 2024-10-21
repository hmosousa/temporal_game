from typing import Dict

from src.base import Timeline


def accuracy(predicted_timeline: Timeline, true_timeline: Timeline) -> float:
    if len(true_timeline) == 0:
        return 1.0

    intersection = true_timeline & predicted_timeline
    return len(intersection) / len(true_timeline)


def precision(predicted_timeline: Timeline, true_timeline: Timeline) -> float:
    if len(true_timeline) == 0:
        return 1.0

    # get all the entity pairs in the true timeline
    predicted_rel_in_true = Timeline()
    for relation in true_timeline.relations:
        rels = predicted_timeline[relation.source, relation.target]
        for rel in rels:
            predicted_rel_in_true.add(rel)

    if len(predicted_rel_in_true) == 0:
        return 0.0

    result = len(true_timeline & predicted_rel_in_true) / len(predicted_rel_in_true)
    return result


def recall(predicted_timeline: Timeline, true_timeline: Timeline) -> float:
    # get all the entity pairs in the true timeline
    predicted_rel_in_true = Timeline()
    for relation in true_timeline.relations:
        rels = predicted_timeline[relation.source, relation.target]
        for rel in rels:
            predicted_rel_in_true.add(rel)

    if len(true_timeline) == 0:
        return 1.0

    result = len(true_timeline & predicted_rel_in_true) / len(true_timeline)
    return result


def f1(predicted_timeline: Timeline, true_timeline: Timeline) -> float:
    p = precision(predicted_timeline, true_timeline)
    r = recall(predicted_timeline, true_timeline)
    return 2 * p * r / (p + r)


def evaluate(predicted_timeline: Timeline, true_timeline: Timeline) -> Dict[str, float]:
    return {
        "accuracy": accuracy(predicted_timeline, true_timeline),
        "precision": precision(predicted_timeline, true_timeline),
        "recall": recall(predicted_timeline, true_timeline),
        "f1": f1(predicted_timeline, true_timeline),
    }
