import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.base import Relation
from src.env import State, TemporalGame


class Node:
    def __init__(
        self,
        state: State,
        parent: Optional["Node"] = None,
        action: Optional[Relation] = None,
        action_index: Optional[int] = None,
        n_actions: int = 0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.action_index = action_index
        self.is_expanded = False
        self.children: Dict[Relation, "Node"] = {}

        # Use numpy arrays for child statistics
        self.child_total_value = np.zeros(n_actions, dtype=np.float32)
        self.child_visits = np.zeros(n_actions, dtype=np.float32)

    def __repr__(self) -> str:
        return f"Node(action={self.action}, visits={self.n_visits}, value={self.total_value})"

    @property
    def n_visits(self):
        return self.parent.child_visits[self.action_index] if self.parent else 0

    @n_visits.setter
    def n_visits(self, value):
        if self.parent:
            self.parent.child_visits[self.action_index] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action_index] if self.parent else 0

    @total_value.setter
    def total_value(self, value):
        if self.parent:
            self.parent.child_total_value[self.action_index] = value

    def child_Q(self) -> np.ndarray:
        """Vectorized Q value calculation"""
        return self.child_total_value / (1 + self.child_visits)

    def child_U(self, exploration_constant: float = 1.41) -> np.ndarray:
        """Vectorized UCB calculation"""
        return exploration_constant * np.sqrt(
            np.log(self.n_visits) / (1 + self.child_visits)
        )

    def get_ucb_score(self, exploration_constant: float = 1.41) -> np.ndarray:
        """Vectorized UCB calculation"""
        return self.child_Q() + self.child_U(exploration_constant)

    def best_child(self) -> int:
        """Returns index of best child using vectorized operations"""
        return np.argmax(self.get_ucb_score())


class MCTS:
    def __init__(self, n_simulations: int = 100):
        self.n_simulations = n_simulations

    def get_actions(self, state: State, env: TemporalGame) -> List[Relation]:
        """Get all valid actions from the current state."""
        actions = [
            Relation(source=pair["source"], target=pair["target"], type=relation_type)
            for pair in state["entity_pairs"]
            for relation_type in env.relation_types
        ]
        return actions

    def select(self, node: Node, env: TemporalGame) -> Tuple[Node, List[Relation]]:
        """Select a node using UCB"""
        current = node
        actions = self.get_actions(current.state, env)

        while current.is_expanded and actions:
            action_idx = current.best_child()

            action = actions[action_idx]
            if action not in current.children:
                break

            current = current.children[action]
            actions = self.get_actions(current.state, env)

        return current, actions

    def get_actions_to_node(self, node: Node) -> List[Relation]:
        """Get the actions that lead to the node."""
        actions = []
        current = node
        while current.parent is not None:
            actions.append(current.action)
            current = current.parent
        return actions[::-1]

    def get_env_to_node(self, env: TemporalGame, node: Node) -> TemporalGame:
        """Get the environment to the state of the node."""
        previous_actions = self.get_actions_to_node(node)
        for action in previous_actions:
            env.step(action)
        return env

    def expand(
        self, node: Node, valid_actions: List[Relation], env: TemporalGame
    ) -> Tuple[Node, bool, float]:
        """Expand the node by adding a new child."""
        # Choose a random unexplored action
        unexplored_indices = [
            i for i, a in enumerate(valid_actions) if a not in node.children
        ]
        if not unexplored_indices:
            return node, False, 0.0

        action_idx = random.choice(unexplored_indices)
        action = valid_actions[action_idx]

        # Create new state
        new_state, reward, terminated, truncated, info = env.step(action)

        # Create new node
        n_actions = len(self.get_actions(new_state, env))
        child = Node(
            state=new_state,
            parent=node,
            action=action,
            action_index=action_idx,
            n_actions=n_actions,
        )
        node.children[action] = child

        # If terminal state, return the reward directly
        if terminated or truncated:
            max_reward_from_state = info["max_reward"] - env.running_reward
            reward_ratio = (
                reward / max_reward_from_state if max_reward_from_state > 0 else 0
            )
            return child, True, reward_ratio

        return child, False, 0.0

    def simulate(self, state: State, env: TemporalGame) -> float:
        """Run a random simulation from the state."""
        current_state = state
        total_reward = 0

        while True:
            entity_pair = random.choice(current_state["entity_pairs"])
            relation_type = random.choice(env.relation_types)
            action = Relation(
                source=entity_pair["source"],
                target=entity_pair["target"],
                type=relation_type,
            )

            current_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        max_reward_from_state = info["max_reward"] - env.running_reward
        reward_ratio = (
            total_reward / max_reward_from_state if max_reward_from_state > 0 else 0
        )
        return reward_ratio

    def backpropagate(self, node: Node, value: float):
        """Backpropagate the value up the tree."""
        current = node
        while current is not None:
            current.n_visits += 1
            current.total_value += value
            current = current.parent

    def search(self, state: State, env: TemporalGame) -> Relation:
        """Perform MCTS and return the best action."""
        n_actions = len(self.get_actions(state, env))
        root = Node(state=state, n_actions=n_actions)

        for _ in range(self.n_simulations):
            selected_node, actions = self.select(root, env)

            mcts_env = self.get_env_to_node(copy.deepcopy(env), selected_node)

            if actions:
                child, is_terminal, value = self.expand(
                    selected_node, actions, mcts_env
                )
                if not is_terminal:
                    value = self.simulate(child.state, mcts_env)

                self.backpropagate(child, value)

        # Choose action with highest visit count
        visit_counts = root.child_visits
        actions = self.get_actions(state, env)
        return actions[np.argmax(visit_counts)]
