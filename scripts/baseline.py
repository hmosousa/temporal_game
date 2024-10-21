import logging

import numpy as np
from fire import Fire

from src.agents import Agent, load_agent
from src.env import TemporalGame
from src.evaluation import evaluate
from src.utils import store_results


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def test(agent: Agent, logger: logging.Logger):
    env = TemporalGame(test=True)
    results = []
    for i in range(5):  # tqdm(range(env.num_docs)):
        episode_reward = 0
        step_count = 0

        state, info = env.reset(i)
        while True:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        predicted_timeline = state["timeline"]
        true_timeline = info["doc_timeline"]
        episode_result = evaluate(predicted_timeline, true_timeline)
        episode_result["step_count"] = step_count
        episode_result["reward"] = episode_reward
        results.append(episode_result)

    mean_results = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    return mean_results


def main(agent_name: str, model_name: str = None):
    """Run the baseline agent on the test set.

    Args:
        agent_name (str): The name of the agent to run.
        model_name (str, optional): In case the agent is a LM, the name of the model to use. Defaults to None.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    agent = load_agent(agent_name, model_name)
    results = test(agent, logger)

    filename = (
        f"{agent_name}_{model_name}".lower() if model_name else agent_name.lower()
    )
    store_results(results, filename)


if __name__ == "__main__":
    Fire(main)
