# Usage: from src.agents.recurrent_ppo import RecurrentPPOAgent
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.base import BaseAgent, TrainingLogger
from src.environment import InventoryEnv


class RecurrentPPOAgent(BaseAgent):
    """Recurrent PPO agent with LSTM for POMDP inventory control.

    Maintains a learned belief state via LSTM hidden states, enabling
    filtering of delayed/noisy/censored observations over time.
    This is the POMDP-aware agent expected to outperform memoryless methods.
    """

    name = "recurrent_ppo"

    def __init__(self, agent_config: dict, env_config: dict):
        self.agent_config = agent_config
        self.env_config = env_config
        self.model = None

    def train(self, env, total_timesteps: int = None, seed: int = 42, callback=None):
        if total_timesteps is None:
            total_timesteps = self.agent_config.get("total_timesteps", 500000)

        vec_env = DummyVecEnv([lambda: InventoryEnv(self.env_config)])

        pk = self.agent_config.get("policy_kwargs", {})
        policy_kwargs = {
            "lstm_hidden_size": pk.get("lstm_hidden_size", 128),
            "n_lstm_layers": pk.get("n_lstm_layers", 1),
        }
        if "net_arch" in pk:
            policy_kwargs["net_arch"] = dict(
                pi=pk["net_arch"]["pi"], vf=pk["net_arch"]["vf"]
            )

        self.model = RecurrentPPO(
            "MlpLstmPolicy",
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
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0,
        )

        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        vec_env.close()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if episode_start is None:
            episode_start = np.ones((1,), dtype=bool)
        return self.model.predict(
            obs, state=state, episode_start=episode_start, deterministic=deterministic
        )

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model = RecurrentPPO.load(path)
