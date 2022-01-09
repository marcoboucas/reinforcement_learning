"""Basic model."""

from random import randint
from typing import Any

import numpy as np

from src.models.base_model import BaseModel
from src.utils import argmax


class Model(BaseModel):
    """Model."""

    model_based = True

    def __init__(self, action_space, observation_space, env, gamma=0.9):
        """Action space."""
        super().__init__(action_space, observation_space, env)
        self.action_space = action_space
        self.observation_space = observation_space
        self.env = env
        self.gamma = gamma

        # Initialize the policy and value function
        self.policy = {
            state: randint(0, self.action_space.n - 1) for state in range(self.observation_space.n)
        }
        self.V = np.zeros((self.observation_space.n,))

        # Loop and update
        for _ in range(1000):
            values_same = self.update_values()
            policy_same = self.update_policy()
            if values_same and policy_same:
                break

    def __call__(self, observations) -> Any:
        """Get action."""
        return self.policy[observations]

    def update_values(self) -> None:
        """Update the values."""
        current_sum = self.V.sum()
        for state in range(self.observation_space.n):
            self.V[state] = sum(
                [
                    prob * (reward + self.gamma * self.V[next_state])
                    for prob, next_state, reward, done in self.env.env.P[state][self.policy[state]]
                ]
            )
        return abs(current_sum - self.V.sum()) < 1e-4

    def update_policy(self) -> bool:
        """Update the policy."""
        is_same = True
        for state in range(self.observation_space.n):
            new_state_policy = argmax(
                [
                    sum(
                        [
                            prob * (reward + self.gamma * self.V[next_state])
                            for prob, next_state, reward, done in self.env.env.P[state][action]
                        ]
                    )
                    for action in range(self.action_space.n)
                ]
            )
            if self.policy[state] != new_state_policy:
                is_same = False
            self.policy[state] = new_state_policy
        return is_same
