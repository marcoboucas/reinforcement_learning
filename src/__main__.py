"""Main file."""

import logging
from time import sleep
import fire
import gym
import importlib
from tqdm import tqdm

ENV_CONVERSION = {"frozenlake": "FrozenLake-v1"}


def main(
    env: str = "frozenlake",
    model: str = "random",
    episodes: int = 100,
    render: bool = False,
):
    """Main function."""
    # Load the env
    environment = gym.make(ENV_CONVERSION[env])
    observation = environment.reset()
    reward = 0.0

    # Load the model
    model_module = importlib.import_module(f"src.models.{env}.{model}")
    model = getattr(model_module, "Model")(
        environment.action_space, environment.observation_space
    )

    for _ in tqdm(range(episodes), desc="Episodes", unit="episodes", disable=render):
        if render:
            environment.render()
        action_done = model(observation, reward)
        old_observation = observation
        observation, reward, done, info = environment.step(action_done)
        model.update(old_observation, action_done, observation, reward)
        if done:
            environment.reset()
            logging.info("RESET THE ENV")
        sleep(0.01)

    environment.close()


if __name__ == "__main__":
    fire.Fire(main)
