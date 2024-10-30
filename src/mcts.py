import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.base import Relation
from src.env import State, TemporalGame


@dataclass
class Node:
    state: State
    parent: Optional["Node"]
    action: Optional[Relation]  # Action that led to this state
    children: Dict[Relation, "Node"]
    visits: int
    value: float

    def __init__(
        self,
        state: State,
        parent: Optional["Node"] = None,
        action: Optional[Relation] = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def __repr__(self) -> str:
        return f"Node(action={self.action}, visits={self.visits}, value={self.value})"

    def __str__(self) -> str:
        return self.__repr__()

    def is_fully_expanded(self, valid_actions: List[Relation]) -> bool:
        return len(self.children) == len(valid_actions)

    def get_ucb_score(
        self, parent_visits: int, exploration_constant: float = 1.41
    ) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return exploitation + exploration


class MCTS:
    def __init__(self, num_simulations: int = 100):
        self.num_simulations = num_simulations

    def get_actions(self, state: State, env: TemporalGame) -> List[Relation]:
        """Get all valid actions from the current state."""
        actions = [
            Relation(source=pair["source"], target=pair["target"], type=relation_type)
            for pair in state["entity_pairs"]
            for relation_type in env.relation_types
        ]
        return actions

    def select(self, node: Node, env: TemporalGame) -> Tuple[Node, List[Relation]]:
        """Select a node to expand using UCB1."""
        current = node
        actions = self.get_actions(current.state, env)

        while current.is_fully_expanded(actions) and actions:
            best_ucb = float("-inf")
            best_child = None

            for child in current.children.values():
                ucb = child.get_ucb_score(current.visits)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

            if best_child is None:
                break

            current = best_child
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
        """Expand the node by adding a new child.

        Returns:
            Tuple containing:
            - The new node
            - Whether it's a terminal state
            - The reward if it's a terminal state
        """
        # Choose a random unexplored action
        unexplored = [a for a in valid_actions if a not in node.children]
        if not unexplored:
            return node, False, 0.0

        action = random.choice(unexplored)

        # Create new state
        new_state, reward, terminated, truncated, info = env.step(action)

        # Create new node
        child = Node(new_state, parent=node, action=action)
        node.children[action] = child

        # If terminal state, return the reward directly
        if terminated or truncated:
            # Calculate reward ratio for terminal state
            max_reward_from_state = info["max_reward"] - env.running_reward
            reward_ratio = reward / max_reward_from_state
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

        # the percentage of reward obtained from the current state
        max_reward_from_state = info["max_reward"] - env.running_reward
        reward_ratio = total_reward / max_reward_from_state
        return reward_ratio

    def backpropagate(self, node: Node, value: float):
        """Backpropagate the value up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

    def search(self, state: State, env: TemporalGame) -> Relation:
        """Perform MCTS and return the best action."""
        root = Node(state)

        for _ in range(self.num_simulations):
            selected_node, actions = self.select(root, env)

            mcts_env = copy.deepcopy(env)
            mcts_env = self.get_env_to_node(mcts_env, selected_node)

            if actions:
                child, is_terminal, terminal_reward = self.expand(
                    selected_node, actions, mcts_env
                )

                if is_terminal:
                    # If terminal state, use the reward directly
                    value = terminal_reward
                else:
                    # Otherwise, simulate from this state
                    value = self.simulate(child.state, mcts_env)

                self.backpropagate(child, value)

        # Choose the action with the highest number of visits
        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]

        return best_action
