import logging

from src.agents import LMAgentNoContext
from src.env import TemporalGame


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    agent = LMAgentNoContext("meta-llama/Llama-3.2-1B")
    env = TemporalGame()

    logger.info("Starting the game")
    state, _ = env.reset()

    episode_reward = 0
    step_count = 0

    while True:
        action = agent.act(state)
        logger.debug(f"Step {step_count}: Action taken: {action}")

        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        step_count += 1

        logger.debug(f"Step {step_count} Reward: {reward}")

        if terminated or truncated:
            logger.info(f"Episode ended after {step_count} steps")
            logger.info(f"Total reward: {episode_reward}")
            logger.info(
                f"Termination reason: {'Terminated' if terminated else 'Truncated'}"
            )
            break

    logger.info("Game finished")


if __name__ == "__main__":
    main()
