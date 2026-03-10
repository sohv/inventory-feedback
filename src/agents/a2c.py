# Usage: from src.agents.a2c import A2CAgent
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.base import BaseAgent
from src.environment import InventoryEnv


class A2CAgent(BaseAgent):
    """A2C (Advantage Actor-Critic) agent for inventory control."""

    name = "a2c"

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
                policy_kwargs["net_arch"] = dict(
                    pi=pk["net_arch"]["pi"], vf=pk["net_arch"]["vf"]
                )

        self.model = A2C(
            "MlpPolicy",
            vec_env,
            learning_rate=self.agent_config.get("learning_rate", 7e-4),
            n_steps=self.agent_config.get("n_steps", 5),
            gamma=self.agent_config.get("gamma", 0.99),
            gae_lambda=self.agent_config.get("gae_lambda", 1.0),
            ent_coef=self.agent_config.get("ent_coef", 0.0),
            vf_coef=self.agent_config.get("vf_coef", 0.5),
            max_grad_norm=self.agent_config.get("max_grad_norm", 0.5),
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
        self.model = A2C.load(path)
