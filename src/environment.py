# Usage: from src.environment import InventoryEnv
import numpy as np
import gymnasium
from gymnasium import spaces


class InventoryEnv(gymnasium.Env):
    """POMDP Inventory Environment with delayed, noisy, and censored demand feedback.

    State (hidden): true demand rate, true inventory level.
    Observation: delayed/noisy/censored sales data + observed inventory + pending orders.
    Action: discrete order quantity in [0, max_order].
    Reward: negative total cost (holding + stockout + ordering).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        super().__init__()
        self.max_inventory = config.get("max_inventory", 200)
        self.max_order = config.get("max_order", 50)
        self.lead_time = config.get("lead_time", 2)
        self.holding_cost = config.get("holding_cost", 1.0)
        self.stockout_cost = config.get("stockout_cost", 10.0)
        self.ordering_cost = config.get("ordering_cost", 2.0)
        self.demand_mean = config.get("demand_mean", 20)
        self.observation_delay = config.get("observation_delay", 2)
        self.noise_std = config.get("noise_std", 3.0)
        self.censoring = config.get("censoring", True)
        self.episode_length = config.get("episode_length", 200)
        self.history_length = config.get("history_length", 10)
        self.seasonality_amplitude = config.get("seasonality_amplitude", 5.0)
        self.seasonality_period = config.get("seasonality_period", 52)

        self.action_space = spaces.Discrete(self.max_order + 1)
        obs_dim = 1 + self.history_length + self.lead_time
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = float(self.max_inventory // 2)
        self.step_count = 0
        self.true_demands = []
        self.inventory_history = []
        self.observed_demands = []
        self.pending_orders = []
        self.total_cost = 0.0
        self.total_lost_sales = 0
        self.total_demand = 0
        self.total_sales = 0
        return self._get_obs(), self._get_info()

    def _generate_demand(self):
        seasonal = self.seasonality_amplitude * np.sin(
            2 * np.pi * self.step_count / self.seasonality_period
        )
        rate = max(1.0, self.demand_mean + seasonal)
        return int(self.np_random.poisson(rate))

    def step(self, action):
        order_qty = int(action)

        new_pending = []
        for qty, arrival_step in self.pending_orders:
            if arrival_step <= self.step_count:
                self.inventory += qty
            else:
                new_pending.append((qty, arrival_step))
        self.pending_orders = new_pending

        self.inventory = min(self.inventory, float(self.max_inventory))

        # Store inventory available to satisfy demand (for censoring computation)
        self.inventory_history.append(self.inventory)

        demand = self._generate_demand()
        self.true_demands.append(demand)

        available = max(0.0, self.inventory)
        sales = min(float(demand), available)
        lost_sales = demand - sales
        self.inventory -= sales

        self.total_demand += demand
        self.total_sales += sales
        self.total_lost_sales += lost_sales

        obs_target = self.step_count - self.observation_delay
        if obs_target >= 0 and obs_target < len(self.true_demands):
            true_d = self.true_demands[obs_target]
            if self.censoring and obs_target < len(self.inventory_history):
                inv_at_obs = max(0.0, self.inventory_history[obs_target])
                censored_d = min(float(true_d), inv_at_obs)
                obs_d = censored_d + self.np_random.normal(0, self.noise_std)
            else:
                obs_d = float(true_d) + self.np_random.normal(0, self.noise_std)
            self.observed_demands.append(max(0.0, obs_d))

        if order_qty > 0:
            self.pending_orders.append(
                (float(order_qty), self.step_count + self.lead_time)
            )

        holding = self.holding_cost * max(self.inventory, 0.0)
        stockout = self.stockout_cost * lost_sales
        ordering = self.ordering_cost * order_qty
        step_cost = holding + stockout + ordering
        reward = -step_cost
        self.total_cost += step_cost

        self.step_count += 1
        truncated = self.step_count >= self.episode_length
        terminated = False

        info = {
            "inventory": self.inventory,
            "step": self.step_count,
            "demand": demand,
            "sales": sales,
            "lost_sales": lost_sales,
            "order": order_qty,
            "holding_cost": holding,
            "stockout_cost": stockout,
            "ordering_cost": ordering,
            "step_cost": step_cost,
            "total_cost": self.total_cost,
            "service_level": self.total_sales / max(1, self.total_demand),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs_inv = np.clip(self.inventory / self.max_inventory, -2.0, 2.0)

        demand_obs = np.zeros(self.history_length, dtype=np.float32)
        n = len(self.observed_demands)
        for i in range(min(self.history_length, n)):
            demand_obs[i] = self.observed_demands[n - 1 - i] / (
                2 * self.demand_mean + 1e-8
            )
        demand_obs = np.clip(demand_obs, -2.0, 2.0)

        pending = np.zeros(self.lead_time, dtype=np.float32)
        for qty, arrival_step in self.pending_orders:
            idx = arrival_step - self.step_count
            if 0 <= idx < self.lead_time:
                pending[idx] += qty / (self.max_order + 1e-8)
        pending = np.clip(pending, 0.0, 2.0)

        return np.concatenate([[obs_inv], demand_obs, pending]).astype(np.float32)

    def _get_info(self):
        return {
            "inventory": self.inventory,
            "step": self.step_count,
        }

    def get_config(self):
        return {
            "max_inventory": self.max_inventory,
            "max_order": self.max_order,
            "lead_time": self.lead_time,
            "holding_cost": self.holding_cost,
            "stockout_cost": self.stockout_cost,
            "ordering_cost": self.ordering_cost,
            "demand_mean": self.demand_mean,
            "observation_delay": self.observation_delay,
            "noise_std": self.noise_std,
            "censoring": self.censoring,
            "episode_length": self.episode_length,
            "history_length": self.history_length,
            "seasonality_amplitude": self.seasonality_amplitude,
            "seasonality_period": self.seasonality_period,
        }
