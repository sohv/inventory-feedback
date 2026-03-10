# Usage: from src.agents.base import BaseAgent
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLogger(BaseCallback):
    """Logs training metrics during RL training."""

    def __init__(self, log_freq: int = 1000):
        super().__init__()
        self.log_freq = log_freq
        self.timesteps = []
        self.episode_rewards = []
        self.losses = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self.timesteps.append(self.num_timesteps)
            if self.model.ep_info_buffer:
                mean_ep_reward = self.model.ep_info_buffer.mean()
                self.episode_rewards.append(float(mean_ep_reward))
            if hasattr(self.model, 'logger') and self.model.logger:
                self.losses.append(None)
        return True

    def save_logs(self, filepath: str):
        """Save logs to JSON file."""
        logs = {
            "timesteps": self.timesteps,
            "episode_rewards": self.episode_rewards,
            "losses": self.losses,
        }
        with open(filepath, 'w') as f:
            json.dump(logs, f)


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
