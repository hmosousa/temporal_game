import json
import logging
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np
from fire import Fire

from src.agents import Agent, load_agent
from src.constants import CACHE_DIR, LOGS_DIR, RESULTS_DIR
from src.env import TemporalGame
from src.evaluation import evaluate
from tqdm import tqdm


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def _save_dict(results: Dict, opath: Path):
    opath.parent.mkdir(exist_ok=True, parents=True)
    with open(opath, "w") as f:
        json.dump(results, f, indent=4)


def store_results(results: Dict, agent_name: str):
    opath = RESULTS_DIR / "agents" / f"{agent_name}.json"
    _save_dict(results, opath)


def cache_results(results: Dict, agent_name: str, doc_id: int):
    opath = CACHE_DIR / "agents" / f"{agent_name}" / f"{doc_id}.json"
    _save_dict(results, opath)


def _save_logs(logs: Dict, agent_name: str, episode_id: int):
    opath = LOGS_DIR / "agents" / f"{agent_name}" / f"{episode_id}.json"
    opath.parent.mkdir(exist_ok=True, parents=True)
    _save_dict(logs, opath)


def test_one_episode(episode_id, agent, env, logger):
    episode_reward = 0
    step_count = 0

    state, info = env.reset(episode_id)
    logger.debug(f"Starting episode {episode_id+1}")

    annotated_timeline = info["doc_timeline"].to_dict()
    episode_logs = {
        "context": state["context"],
        "annotated_timeline": annotated_timeline["relations"],
        "entities": annotated_timeline["entities"],
        "steps": [
            {
                "step": 0,
                "action": None,
                "timeline": state["timeline"].to_dict()["relations"],
                "reward": 0,
            }
        ],
    }

    while True:
        if agent.name == "mcts":
            action = agent.act(state, env)
        else:
            action = agent.act(state)

        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        step_count += 1

        episode_logs["steps"].append(
            {
                "step": step_count,
                "action": action.to_dict(),
                "timeline": state["timeline"].to_dict()["relations"],
                "reward": reward,
            }
        )

        if terminated or truncated:
            break

    _save_logs(episode_logs, agent.name, episode_id)

    predicted_timeline = state["timeline"]
    true_timeline = info["doc_timeline"]

    episode_result = evaluate(predicted_timeline, true_timeline)
    episode_result["step_count"] = step_count
    episode_result["reward"] = episode_reward

    # make copy of episode_result to add extra info
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

    cache_info = {
        "context": state["context"],
        "timeline": annot_diff,
        "result": episode_result,
    }
    cache_results(cache_info, agent.name, episode_id)

    logger.debug(
        f"Episode {episode_id+1} completed. Reward: {episode_reward}, Steps: {step_count}"
    )
    return episode_result


def test_in_parallel(env: TemporalGame, agent: Agent, logger: logging.Logger):
    thread_local = threading.local()

    def get_env():
        if not hasattr(thread_local, "env"):
            thread_local.env = TemporalGame(test=True)
        return thread_local.env

    def run_episode(episode_id):
        thread_env = get_env()
        return test_one_episode(episode_id, agent, thread_env, logger)

    results = []
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(
            tqdm(executor.map(run_episode, range(env.num_docs)), total=env.num_docs)
        )

    return results


def test_sequential(env: TemporalGame, agent: Agent, logger: logging.Logger):
    results = []
    for episode_id in tqdm(range(env.num_docs)):
        results.append(test_one_episode(episode_id, agent, env, logger))
    return results


def test(agent: Agent, logger: logging.Logger, in_parallel: bool = True):
    env = TemporalGame(test=True)
    logger.info(f"Starting test with {env.num_docs} documents")

    if in_parallel:
        results = test_in_parallel(env, agent, logger)
    else:
        results = test_sequential(env, agent, logger)

    mean_results = {
        k: float(np.mean([r[k] for r in results]).round(4)) for k in results[0].keys()
    }

    logger.info(f"Test completed. Mean results: {mean_results}")
    return mean_results


def main(
    agent_name: str = "before",
    model_name: str = None,
    num_simulations: int = None,
    in_parallel: bool = False,
):
    """Run the baseline agent on the test set.

    Args:
        agent_name (str): The name of the agent to run.
        model_name (str, optional): In case the agent is a LM, the name of the model to use. Defaults to None.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    agent = load_agent(agent_name, model_name, num_simulations)
    results = test(agent, logger, in_parallel)

    filename = (
        f"{agent_name}_{model_name}".lower() if model_name else agent_name.lower()
    )
    store_results(results, filename)


if __name__ == "__main__":
    Fire(main)
