# Usage: from src.agents.base import BaseAgent
import numpy as np


class BaseAgent:
    """Base class for all inventory agents."""

    name: str = "base"

    def predict(self, obs: np.ndarray, state=None, episode_start=None, deterministic=True):
        raise NotImplementedError

    def train(self, env, total_timesteps: int, seed: int = 42):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
