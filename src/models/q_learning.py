"""Basic model."""

from random import randint
from typing import Any

import numpy as np

from src.models.base_model import BaseModel
from src.utils import argmax


# pylint: disable=too-many-instance-attributes,too-many-arguments
class Model(BaseModel):
    """Model."""

    model_based = False

    def __init__(
        self,
        action_space,
        observation_space,
        env,
        gamma=0.99,
        alpha=0.5,
        exploration_rate=1.0,
        exploration_rate_decay=0.99,
    ):
        """Action space."""
        super().__init__(action_space, observation_space, env)
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.alpha = alpha
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.learning_updates = 0

        # Initialize the policy and value function
        self.Q = np.zeros((self.observation_space.n, self.action_space.n))

    def __call__(self, observations) -> Any:
        """Get action."""
        if self.learning_updates % 10000 == 0 and self.exploration_rate > 0.1:
            self.exploration_rate *= self.exploration_rate_decay
        return (
            argmax(self.Q[observations])
            if np.random.random() > self.exploration_rate
            else randint(0, self.action_space.n - 1)
        )

    def update(self, old_observation, action_done, observation, reward):
        """Update the quality function."""
        self.Q[old_observation, action_done] += (
            self.alpha
            * (
                reward
                + self.gamma
                * max(
                    [self.Q[observation, next_action] for next_action in range(self.action_space.n)]
                )
            )
            - self.Q[old_observation, action_done]
        )
        self.learning_updates += 1
