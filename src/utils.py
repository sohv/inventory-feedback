# Usage: from src.utils import load_config, evaluate_agent, set_seed
import json
from pathlib import Path

import numpy as np
import yaml

from src.environment import InventoryEnv

RESULTS_DIR = Path(__file__).parent / "results"


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_env(env_config: dict, seed: int = 42) -> InventoryEnv:
    env = InventoryEnv(env_config)
    env.reset(seed=seed)
    return env


def evaluate_agent(agent, env_config: dict, num_episodes: int = 50, seed: int = 1000):
    """Evaluate an agent and return detailed metrics.

    Works with both SB3 models and custom agents (SSPolicy, MPCAgent).
    """
    metrics = {
        "episode_costs": [],
        "episode_service_levels": [],
        "episode_stockout_steps": [],
        "episode_rewards": [],
    }

    for ep in range(num_episodes):
        env = InventoryEnv(env_config)
        obs, info = env.reset(seed=seed + ep)

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        total_reward = 0.0
        stockout_steps = 0

        done = False
        while not done:
            try:
                action, lstm_states = agent.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
            except TypeError:
                action, _ = agent.predict(obs, deterministic=True)

            if isinstance(action, np.ndarray):
                action = action.item() if action.ndim == 0 else action[0]

            obs, reward, terminated, truncated, info = env.step(action)
            episode_starts = np.array([terminated or truncated])
            total_reward += reward

            if info.get("lost_sales", 0) > 0:
                stockout_steps += 1

            done = terminated or truncated

        metrics["episode_costs"].append(info.get("total_cost", -total_reward))
        metrics["episode_service_levels"].append(info.get("service_level", 0.0))
        metrics["episode_stockout_steps"].append(stockout_steps)
        metrics["episode_rewards"].append(total_reward)

    summary = {
        "mean_cost": float(np.mean(metrics["episode_costs"])),
        "std_cost": float(np.std(metrics["episode_costs"])),
        "mean_reward": float(np.mean(metrics["episode_rewards"])),
        "std_reward": float(np.std(metrics["episode_rewards"])),
        "mean_service_level": float(np.mean(metrics["episode_service_levels"])),
        "std_service_level": float(np.std(metrics["episode_service_levels"])),
        "mean_stockout_steps": float(np.mean(metrics["episode_stockout_steps"])),
        "stockout_rate": float(
            np.mean(metrics["episode_stockout_steps"])
            / env_config.get("episode_length", 200)
        ),
    }
    return summary


def save_results(results: dict, filename: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filename: str) -> dict:
    filepath = RESULTS_DIR / filename
    with open(filepath) as f:
        return json.load(f)


def make_env_config(base_config: dict, **overrides) -> dict:
    config = dict(base_config)
    config.update(overrides)
    return config
