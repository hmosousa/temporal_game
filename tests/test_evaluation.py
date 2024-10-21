import pytest

from src.base import Relation, Timeline
from src.evaluation import accuracy, f1, precision, recall


@pytest.fixture
def sample_timelines():
    true_timeline = Timeline([Relation("A", "B", "<"), Relation("B", "C", "<")])
    predicted_timeline = Timeline(
        [Relation("A", "B", "<"), Relation("B", "C", ">"), Relation("C", "E", "<")]
    )
    return true_timeline, predicted_timeline


def test_accuracy(sample_timelines):
    true_timeline, predicted_timeline = sample_timelines
    accuracy_score = accuracy(predicted_timeline, true_timeline)
    assert accuracy_score == pytest.approx(0.5)


def test_precision(sample_timelines):
    true_timeline, predicted_timeline = sample_timelines
    precision_score = precision(predicted_timeline, true_timeline)
    assert precision_score == pytest.approx(0.5)


def test_recall(sample_timelines):
    true_timeline, predicted_timeline = sample_timelines
    recall_score = recall(predicted_timeline, true_timeline)
    assert recall_score == pytest.approx(0.5)


def test_f1(sample_timelines):
    true_timeline, predicted_timeline = sample_timelines
    f1_score = f1(predicted_timeline, true_timeline)
    assert f1_score == pytest.approx(0.5)


def test_empty_timelines():
    empty_timeline = Timeline([])
    assert accuracy(empty_timeline, empty_timeline) == 1.0
    assert precision(empty_timeline, empty_timeline) == 1.0
    assert recall(empty_timeline, empty_timeline) == 1.0
    assert f1(empty_timeline, empty_timeline) == 1.0
