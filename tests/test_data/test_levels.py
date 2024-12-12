import pytest

from src.data import load_levels


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_levels_0():
    trainset = load_levels("train", level=0)
    assert len(trainset) == 311_900

    testset = load_levels("test", level=0)
    assert len(testset) == 9_604


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_levels_1():
    trainset = load_levels("train", level=1)
    assert len(trainset) == 311_900

    testset = load_levels("test", level=1)
    assert len(testset) == 9_604


@pytest.mark.skip(reason="Skipping due to slow loading times")
def test_load_levels_2():
    trainset = load_levels("train", level=2)
    assert len(trainset) == 311_900

    testset = load_levels("test", level=2)
    assert len(testset) == 9_604
