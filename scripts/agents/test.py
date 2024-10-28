import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from fire import Fire

from src.agents import Agent, load_agent
from src.constants import CACHE_DIR, RESULTS_DIR
from src.env import TemporalGame
from src.evaluation import evaluate
from tqdm import tqdm


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def _save_dict(results: Dict, opath: Path):
    opath.parent.mkdir(exist_ok=True)
    with open(opath, "w") as f:
        json.dump(results, f, indent=4)


def store_results(results: Dict, agent_name: str):
    opath = RESULTS_DIR / "agents" / f"{agent_name}.json"
    _save_dict(results, opath)


def cache_results(results: Dict, agent_name: str, doc_id: int):
    opath = CACHE_DIR / "agents" / f"{agent_name}" / f"{doc_id}.json"
    _save_dict(results, opath)


def test(agent: Agent, logger: logging.Logger):
    env = TemporalGame(test=True)
    results = []
    logger.info(f"Starting test with {env.num_docs} documents")
    for i in tqdm(range(env.num_docs)):
        episode_reward = 0
        step_count = 0

        state, info = env.reset(i)
        logger.debug(f"Starting episode {i+1}")

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

        # make copy of episode_result to add extra info
        episode_result = episode_result.copy()
        episode_result["context"] = state["context"]

        annot_diff = []
        for true_rel in true_timeline.relations:
            pred_rels = predicted_timeline[true_rel.source, true_rel.target]
            if len(pred_rels) == 0:
                annot_diff.append(
                    {
                        "source": true_rel.source,
                        "target": true_rel.target,
                        "true": true_rel.type,
                        "pred": None,
                    }
                )
            else:
                for pred_rel in pred_rels:
                    annot_diff.append(
                        {
                            "source": true_rel.source,
                            "target": true_rel.target,
                            "true": true_rel.type,
                            "pred": pred_rel.type,
                        }
                    )

        episode_result["timeline"] = annot_diff

        cache_results(episode_result, agent.name, i)

        logger.debug(
            f"Episode {i+1} completed. Reward: {episode_reward}, Steps: {step_count}"
        )

    mean_results = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    logger.info(f"Test completed. Mean results: {mean_results}")
    return mean_results


def main(agent_name: str = "trained", model_name: str = None):
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
