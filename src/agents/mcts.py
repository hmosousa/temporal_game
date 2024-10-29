from src.agents.base import Agent
from src.base import Relation
from src.env import State, TemporalGame
from src.mcts import MCTS


class MCTSAgent(Agent):
    def __init__(self, num_simulations: int = 100):
        self.mcts = MCTS(num_simulations)

    @property
    def name(self) -> str:
        return "mcts"

    def act(self, state: State, env: TemporalGame) -> Relation:
        return self.mcts.search(state, env)
