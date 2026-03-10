# Usage: from src.agents.ppo import PPOAgent
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.base import BaseAgent, TrainingLogger
from src.environment import InventoryEnv


class PPOAgent(BaseAgent):
    """PPO agent for inventory control (memoryless — treats POMDP as MDP)."""

    name = "ppo"

    def __init__(self, agent_config: dict, env_config: dict):
        self.agent_config = agent_config
        self.env_config = env_config
        self.model = None

    def train(self, env, total_timesteps: int = None, seed: int = 42, callback=None):
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

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.agent_config.get("learning_rate", 3e-4),
            n_steps=self.agent_config.get("n_steps", 2048),
            batch_size=self.agent_config.get("batch_size", 64),
            n_epochs=self.agent_config.get("n_epochs", 10),
            gamma=self.agent_config.get("gamma", 0.99),
            gae_lambda=self.agent_config.get("gae_lambda", 0.95),
            clip_range=self.agent_config.get("clip_range", 0.2),
            ent_coef=self.agent_config.get("ent_coef", 0.01),
            vf_coef=self.agent_config.get("vf_coef", 0.5),
            max_grad_norm=self.agent_config.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            seed=seed,
            verbose=0,
        )

        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        vec_env.close()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path)
