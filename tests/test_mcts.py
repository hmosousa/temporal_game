import numpy as np
import pytest
from src.base import Relation
from src.env import TemporalGame

from src.mcts import MCTS, Node


@pytest.fixture
def env():
    return TemporalGame(test=True)


@pytest.fixture
def mcts():
    return MCTS(n_simulations=100)


def test_node_initialization():
    state = {"context": "", "entity_pairs": [], "timeline": []}
    node = Node(state)
    assert node.state == state
    assert node.parent is None
    assert node.action is None
    assert node.children == {}
    assert node.n_visits == 0
    assert node.total_value == 0.0


def test_node_ucb_score():
    state = {"context": "", "entity_pairs": [], "timeline": []}
    parent = Node(state, n_actions=1)
    node = Node(state, parent=parent, n_actions=1)
    parent.children[Relation("start A", "start B", "<")] = node
    node.n_visits = 10
    node.total_value = 5.0
    parent.n_visits = 20
    parent.total_value = 10.0

    ucb_score = node.get_ucb_score()
    assert isinstance(ucb_score, np.ndarray)
    assert ucb_score.shape == (1, 1)
    assert ucb_score[0, 0] > 0


def test_mcts_get_actions(env, mcts):
    state, _ = env.reset(0)
    actions = mcts.get_actions(state, env)

    assert isinstance(actions, list)
    assert all(isinstance(action, Relation) for action in actions)


def test_mcts_select(env, mcts):
    state, _ = env.reset(0)
    root = Node(state)

    selected_node, valid_actions = mcts.select(root, env)
    assert isinstance(selected_node, Node)
    assert isinstance(valid_actions, list)


def test_mcts_expand(env, mcts):
    state, _ = env.reset(0)
    root = Node(state)
    valid_actions = mcts.get_actions(state, env)

    child, is_terminal, reward = mcts.expand(root, valid_actions, env)
    assert isinstance(child, Node)
    assert isinstance(is_terminal, bool)
    assert isinstance(reward, float)


def test_mcts_simulate(env, mcts):
    state, _ = env.reset(0)
    reward = mcts.simulate(state, env)
    assert isinstance(reward, float)
    assert 0 <= reward <= 1  # Reward should be normalized


def test_mcts_backpropagate(env, mcts):
    state, _ = env.reset(0)
    root = Node(state, n_actions=1)
    child = Node(state, parent=root, n_actions=1)
    root.children[Relation("start A", "start B", "<")] = child

    mcts.backpropagate(child, 1.0)

    assert root.n_visits == 0
    assert child.n_visits == np.array([[1]])


def test_mcts_search(env, mcts):
    state, _ = env.reset(0)
    action = mcts.search(state, env)
    assert isinstance(action, Relation)


def test_mcts_with_terminal_state(env, mcts):
    state, _ = env.reset(0)
    root = Node(state)
    valid_actions = mcts.get_actions(state, env)

    # Force a terminal state by making an invalid move
    env.step(valid_actions[0])  # Make first move

    # Expand should handle terminal state
    child, is_terminal, reward = mcts.expand(root, valid_actions, env)
    assert isinstance(child, Node)
    assert isinstance(is_terminal, bool)
    assert isinstance(reward, float)
