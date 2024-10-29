import pytest

from src.agents.mcts import MCTS, Node
from src.base import Relation
from src.env import TemporalGame


@pytest.fixture
def env():
    return TemporalGame(test=True)


@pytest.fixture
def mcts():
    return MCTS(num_simulations=100)


def test_node_initialization():
    state = {"context": "", "entity_pairs": [], "timeline": []}
    node = Node(state)
    assert node.state == state
    assert node.parent is None
    assert node.action is None
    assert node.children == {}
    assert node.visits == 0
    assert node.value == 0.0


def test_node_ucb_score():
    state = {"context": "", "entity_pairs": [], "timeline": []}
    node = Node(state)
    node.visits = 10
    node.value = 5.0
    parent_visits = 20

    ucb_score = node.get_ucb_score(parent_visits)
    assert isinstance(ucb_score, float)
    assert ucb_score > 0


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
    root = Node(state)
    child = Node(state, parent=root)
    root.children[Relation("start A", "start B", "<")] = child

    initial_root_visits = root.visits
    initial_child_visits = child.visits

    mcts.backpropagate(child, 1.0)

    assert root.visits == initial_root_visits + 1
    assert child.visits == initial_child_visits + 1


def test_mcts_search(env, mcts):
    state, _ = env.reset(0)
    action = mcts.search(state, env)
    assert isinstance(action, Relation)


def test_mcts_act(env, mcts):
    state, _ = env.reset(0)
    action = mcts.act(state, env)
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
