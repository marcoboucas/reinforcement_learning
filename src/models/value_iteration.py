"""Basic model."""

from pprint import pprint
from typing import Any

import numpy as np

from src.models.base_model import BaseModel
from src.utils import argmax, vprint


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

        # Initialize the values
        self.V = np.zeros(self.observation_space.n)

        # Implement the loop
        for _ in range(700):
            newV = np.zeros_like(self.V)
            for current_state in range(self.observation_space.n):
                # current_value = self.V[current_state]
                # max (over all states) of: reward + proba * V(new_state)
                newV[current_state] = max(
                    [
                        sum(
                            [
                                prob * (reward + self.gamma * self.V[next_state])
                                for prob, next_state, reward, done in env.P[current_state][action]
                            ]
                        )
                        for action in range(self.action_space.n)
                    ]
                )
            if abs(self.V - newV).sum() < 1e-4:
                break
            self.V = newV
        vprint(self.V, 4)

        self.policy = {}
        for current_state in range(self.observation_space.n):
            scores = [
                sum(
                    [
                        proba * (reward + self.gamma * self.V[future_state])
                        for proba, future_state, reward, done in self.env.env.P[current_state][
                            action
                        ]
                    ]
                )
                for action in range(self.action_space.n)
            ]
            self.policy[current_state] = scores
        pprint({k: np.argmax(v) for k, v in self.policy.items()})

    def __call__(self, observations) -> Any:
        """Get action."""
        return argmax(self.policy[observations])
