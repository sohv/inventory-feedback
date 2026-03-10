# Usage: from src.agents.mpc import MPCAgent
import numpy as np
from scipy.stats import poisson

from src.agents.base import BaseAgent


class MPCAgent(BaseAgent):
    """Model Predictive Control agent using newsvendor-based rolling horizon.

    Forecasts demand from observed history and computes order-up-to level
    using the critical ratio. Degrades when observations are delayed/noisy.
    """

    name = "mpc"

    def __init__(self, agent_config: dict, env_config: dict):
        self.horizon = agent_config.get("horizon", 10)
        self.holding_cost = env_config.get("holding_cost", 1.0)
        self.stockout_cost = env_config.get("stockout_cost", 10.0)
        self.ordering_cost = env_config.get("ordering_cost", 2.0)
        self.lead_time = env_config.get("lead_time", 2)
        self.max_order = env_config.get("max_order", 50)
        self.max_inventory = env_config.get("max_inventory", 200)
        self.demand_mean = env_config.get("demand_mean", 20)
        self.history_length = env_config.get("history_length", 10)

    def predict(self, obs: np.ndarray, state=None, episode_start=None, deterministic=True):
        inventory = obs[0] * self.max_inventory

        demand_history = obs[1 : 1 + self.history_length] * (2 * self.demand_mean)
        valid_demands = demand_history[demand_history > 0.5]

        if len(valid_demands) >= 2:
            demand_est = float(np.mean(valid_demands))
        else:
            demand_est = float(self.demand_mean)

        pending_start = 1 + self.history_length
        pending_raw = obs[pending_start : pending_start + self.lead_time]
        pipeline = float(np.sum(pending_raw * self.max_order))

        inv_position = inventory + pipeline

        critical_ratio = self.stockout_cost / (self.stockout_cost + self.holding_cost)
        demand_over_lt = demand_est * (self.lead_time + 1)
        target_level = poisson.ppf(critical_ratio, max(1.0, demand_over_lt))

        order = int(np.clip(target_level - inv_position, 0, self.max_order))
        return np.array([order]), state
