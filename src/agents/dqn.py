# Usage: from src.agents.dqn import DQNAgent
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.base import BaseAgent
from src.environment import InventoryEnv


class DQNAgent(BaseAgent):
    """DQN agent for inventory control (memoryless — treats POMDP as MDP)."""

    name = "dqn"

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

        self.model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=self.agent_config.get("learning_rate", 1e-4),
            buffer_size=self.agent_config.get("buffer_size", 500000),
            learning_starts=self.agent_config.get("learning_starts", 10000),
            batch_size=self.agent_config.get("batch_size", 64),
            gamma=self.agent_config.get("gamma", 0.99),
            exploration_fraction=self.agent_config.get("exploration_fraction", 0.15),
            exploration_final_eps=self.agent_config.get("exploration_final_eps", 0.02),
            target_update_interval=self.agent_config.get(
                "target_update_interval", 10000
            ),
            train_freq=self.agent_config.get("train_freq", 4),
            gradient_steps=self.agent_config.get("gradient_steps", 1),
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
        self.model = DQN.load(path)
