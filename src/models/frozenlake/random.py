"""Random model."""

from typing import Any

from src.models.base_model import BaseModel

class Model(BaseModel):
    """Model."""

    def __init__(self, action_space, observation_space):
        """Action space."""
        self.action_space = action_space
        self.observation_space = observation_space
    
    def __call__(self, observations, reward) -> Any:
        """Get action."""
        print(observations, type(observations))

        return self.action_space.sample()