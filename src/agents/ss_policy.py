# Usage: from src.agents.ss_policy import SSPolicy
import numpy as np

from src.agents.base import BaseAgent


class SSPolicy(BaseAgent):
    """Classical (s, S) inventory policy.

    Orders up to S when inventory position drops below s.
    Assumes full observability — degrades under noisy/delayed feedback.
    """

    name = "ss_policy"

    def __init__(self, agent_config: dict, env_config: dict):
        self.s = agent_config.get("s", 40)
        self.S = agent_config.get("S", 80)
        self.max_inventory = env_config.get("max_inventory", 200)
        self.max_order = env_config.get("max_order", 50)
        self.lead_time = env_config.get("lead_time", 2)
        self.history_length = env_config.get("history_length", 10)

    def predict(self, obs: np.ndarray, state=None, episode_start=None, deterministic=True):
        inventory = obs[0] * self.max_inventory

        pending_start = 1 + self.history_length
        pending_raw = obs[pending_start : pending_start + self.lead_time]
        pipeline = float(np.sum(pending_raw * self.max_order))

        inv_position = inventory + pipeline

        if inv_position < self.s:
            order = int(np.clip(self.S - inv_position, 0, self.max_order))
        else:
            order = 0

        return np.array([order]), state
