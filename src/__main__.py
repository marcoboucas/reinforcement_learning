"""Main file."""

import importlib
import logging
from time import sleep

import fire
import gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

ENV_CONVERSION = {"frozenlake": "FrozenLake-v1"}


# pylint: disable=too-many-arguments,too-many-locals
def main(
    env: str = "frozenlake",
    model: str = "random",
    episodes: int = 100,
    max_steps: int = -1,
    render: bool = False,
    verbose: bool = False,
    graphs: bool = True,
):
    """Main function."""
    # Load the env
    environment = gym.make(ENV_CONVERSION[env], map_name="4x4")
    observation = environment.reset()
    reward = 0.0

    # Load the model
    model_class = getattr(importlib.import_module(f"src.models.{model}"), "Model")
    model = model_class(
        environment.action_space,
        environment.observation_space,
        environment if model_class.model_based else None,
    )

    results = []
    for _ in tqdm(range(episodes), desc="Episodes", unit="episodes", disable=render):

        observation = environment.reset()
        reward = 0.0

        step = 0
        while True:
            if render:
                environment.render()
                sleep(0.01)
                print("\n")
            action_done = model(observation)
            old_observation = observation
            observation, reward, done, _ = environment.step(action_done)
            model.update(old_observation, action_done, observation, reward)

            if done or step >= max_steps > 0:
                if verbose:
                    environment.render()
                    logging.info("RESET THE ENV (after %d steps): final reward: %d", step, reward)
                    logging.debug(
                        "Old Observation: %s, Action taken: %s, New Observation: %s",
                        old_observation,
                        action_done,
                        observation,
                    )
                results.append({"steps": step, "reward": reward})
                break
            step += 1

    environment.close()

    results_df = pd.DataFrame.from_records(results)
    print(results_df.describe())

    if graphs:
        _, axes = plt.subplots(1, 2)
        sns.histplot(x=results_df["reward"], ax=axes[0])
        axes[0].set_title("Rewards")

        sns.kdeplot(x=results_df["steps"], hue=results_df["reward"], ax=axes[1])
        axes[1].set_title("Number of steps to reach the end or die")

        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
