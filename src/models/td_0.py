"""Basic model."""

from random import randint
from typing import Any

import numpy as np

from src.models.base_model import BaseModel


class Model(BaseModel):
    """Model."""

    model_based = False

    # pylint: disable=too-many-arguments
    def __init__(self, action_space, observation_space, env, gamma=0.9, alpha=0.1):
        """Action space."""
        super().__init__(action_space, observation_space, env)
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.alpha = alpha

        # Initialize the policy and value function
        self.policy = {
            state: randint(0, self.action_space.n - 1) for state in range(self.observation_space.n)
        }
        self.V = np.zeros((self.observation_space.n,))

    def __call__(self, observations) -> Any:
        """Get action."""
        return self.policy[observations]

    def update(self, old_observation, action_done, observation, reward):
        """Update the value function, ability to compute the value function for a given policy."""
        self.V[old_observation] = self.V[old_observation] + self.alpha * (
            reward + self.gamma * self.V[observation] - self.V[old_observation]
        )
