# Usage: from src.agents.trpo import TRPOAgent
from pathlib import Path

from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.base import BaseAgent
from src.environment import InventoryEnv


class TRPOAgent(BaseAgent):
    """Trust Region Policy Optimization (TRPO) agent for inventory control."""

    name = "trpo"

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

        self.model = TRPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.agent_config.get("learning_rate", 1e-3),
            n_steps=self.agent_config.get("n_steps", 2048),
            batch_size=self.agent_config.get("batch_size", 128),
            gamma=self.agent_config.get("gamma", 0.99),
            gae_lambda=self.agent_config.get("gae_lambda", 0.98),
            cg_max_steps=self.agent_config.get("cg_max_steps", 15),
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
        self.model = TRPO.load(path)
