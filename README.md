# Inventory Management under Delayed & Noisy Demand Feedback

A POMDP-based inventory control comparison framework. Evaluates classical baselines (s,S policy, MPC), feedforward RL agents (DQN, PPO), and recurrent RL (RecurrentPPO with LSTM) under delayed, noisy, and censored demand observations.

## Quick Start

### Setup

```bash
# Install dependencies (requires Python 3.10+)
uv sync
```

### Run Experiments

```bash
# Full benchmark (all 3 experiments)
python -m src.experiments.run_benchmark

# Delay sweep only
python -m src.experiments.run_benchmark --experiment delay

# Noise sweep only
python -m src.experiments.run_benchmark --experiment noise

# Censoring comparison only
python -m src.experiments.run_benchmark --experiment censoring
```

### Run Single Agent

```bash
# Train & evaluate a specific agent (replace with dqn, ppo, recurrent_ppo, ss_policy, mpc)
python -m src.experiments.run_single --agent recurrent_ppo

# With custom settings
python -m src.experiments.run_single --agent dqn --seed 42 --eval-episodes 100 --timesteps 750000
```

### Generate Visualizations

```bash
# Generate all plots
python -m src.visualize --plot all

# Specific plots
python -m src.visualize --plot delay
python -m src.visualize --plot noise
python -m src.visualize --plot censoring
python -m src.visualize --plot comparison
python -m src.visualize --plot heatmap
python -m src.visualize --plot advantage
```

**Quick Test Run**: 

```bash
# Fast single agent run (sampling only, no training)
python -m src.experiments.run_single --agent ss_policy --eval-episodes 10
```

## Agents

The framework compares five control policies:

| Agent | Type | Memory | Use Case |
|-------|------|--------|----------|
| (s,S) Policy | Classical OR baseline | No | Full observability baseline |
| MPC | Optimization (rolling horizon) | No | Myopic forecast-based control |
| DQN | Value-based RL | No | Deep Q-learning (memoryless) |
| PPO | Policy gradient RL | No | Proximal policy optimization (memoryless) |
| **RecurrentPPO** | **Policy gradient + LSTM** | **Yes** | **POMDP-aware learning with hidden state** |

## Results

Results are saved as JSON in `src/results/`:
- `delay_sweep.json` - Performance across observation delays
- `noise_sweep.json` - Performance across noise levels  
- `censoring_comparison.json` - Censored vs uncensored demand
- `*_seed*.json` - Individual agent runs

Plots are automatically generated:
- `delay_sweep.png` - Cost/service/stockout vs delay
- `noise_sweep.png` - Cost/service/stockout vs noise
- `censoring_comparison.png` - Censored vs uncensored comparison
- `agent_comparison.png` - Agent comparison on default config
- `robustness_heatmap.png` - Cost heatmaps
- `pomdp_advantage.png` - RecurrentPPO advantage over baselines
