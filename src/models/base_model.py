"""Base Model."""

import logging
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base Model."""

    model_based: bool = False

    @abstractmethod
    def __init__(self, action_space, observation_space, env=None):
        """Init."""
        if env is not None:
            logging.warning("Cheating")

    @abstractmethod
    def __call__(self, observations):
        """Get action."""

    @abstractmethod
    def update(self, old_observation, action_done, observation, reward):
        """Update."""
