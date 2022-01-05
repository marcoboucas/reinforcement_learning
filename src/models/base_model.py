"""Base Model."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base Model."""

    @abstractmethod
    def __init__(self, action_space, observation_space):
        """Init."""

    @abstractmethod
    def __call__(self, observations, reward):
        """Get action."""
    
    @abstractmethod
    def update(self, old_observation, action_done, observation, reward):
        """Update."""