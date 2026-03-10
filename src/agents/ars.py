# Usage: from src.agents.ars import ARSAgent
from pathlib import Path

from sb3_contrib import ARS
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.base import BaseAgent
from src.environment import InventoryEnv


class ARSAgent(BaseAgent):
    """Augmented Random Search (ARS) agent for inventory control."""

    name = "ars"

    def __init__(self, agent_config: dict, env_config: dict):
        self.agent_config = agent_config
        self.env_config = env_config
        self.model = None

    def train(self, env, total_timesteps: int = None, seed: int = 42):
        if total_timesteps is None:
            total_timesteps = self.agent_config.get("total_timesteps", 500000)

        vec_env = DummyVecEnv([lambda: InventoryEnv(self.env_config)])

        policy_kwargs = {}
        if "policy_kwargs" in self.agent_config:
            pk = self.agent_config["policy_kwargs"]
            if "net_arch" in pk:
                policy_kwargs["net_arch"] = pk["net_arch"]

        self.model = ARS(
            "MlpPolicy",
            vec_env,
            learning_rate=self.agent_config.get("learning_rate", 0.03),
            n_delta=self.agent_config.get("n_delta", 16),
            n_top=self.agent_config.get("n_top", 6),
            delta_std=self.agent_config.get("delta_std", 0.02),
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            seed=seed,
            verbose=0,
        )

        self.model.learn(total_timesteps=total_timesteps)
        vec_env.close()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model = ARS.load(path)
