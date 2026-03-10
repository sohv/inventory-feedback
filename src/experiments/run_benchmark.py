# Run: python -m src.experiments.run_benchmark
import argparse
import time

import numpy as np

from src.agents import (
    DQNAgent,
    MPCAgent,
    PPOAgent,
    RecurrentPPOAgent,
    SSPolicy,
)
from src.agents.base import TrainingLogger
from src.environment import InventoryEnv
from src.utils import (
    evaluate_agent,
    load_config,
    make_env_config,
    save_results,
    set_seed,
)

AGENT_CLASSES = {
    "ss_policy": SSPolicy,
    "mpc": MPCAgent,
    "dqn": DQNAgent,
    "ppo": PPOAgent,
    "recurrent_ppo": RecurrentPPOAgent,
}

RL_AGENTS = {"dqn", "ppo", "recurrent_ppo"}


def build_agent(agent_name: str, agent_config: dict, env_config: dict):
    cls = AGENT_CLASSES[agent_name]
    return cls(agent_config, env_config)


def run_single_config(
    env_config: dict,
    agents_config: dict,
    experiment_config: dict,
    config_label: str,
):
    """Train and evaluate all agents on a single environment configuration."""
    num_eval = experiment_config.get("num_eval_episodes", 50)
    num_seeds = experiment_config.get("num_seeds", 3)
    base_seed = experiment_config.get("seed", 42)

    results = {}

    for agent_name in ["ss_policy", "mpc"]:
        print(f"  Evaluating {agent_name}...")
        agent = build_agent(agent_name, agents_config[agent_name], env_config)
        metrics = evaluate_agent(agent, env_config, num_episodes=num_eval, seed=base_seed)
        results[agent_name] = metrics
        print(
            f"    Cost: {metrics['mean_cost']:.1f} +/- {metrics['std_cost']:.1f}  "
            f"Service: {metrics['mean_service_level']:.3f}  "
            f"Stockout rate: {metrics['stockout_rate']:.3f}"
        )

    for agent_name in ["dqn", "ppo", "recurrent_ppo"]:
        seed_results = []
        for s in range(num_seeds):
            seed = base_seed + s * 100
            print(f"  Training {agent_name} (seed {seed})...")
            t0 = time.time()

            agent = build_agent(agent_name, agents_config[agent_name], env_config)
            
            # Setup training logger
            logger = TrainingLogger(log_freq=1000)
            agent.train(None, seed=seed, callback=logger)
            
            # Save training logs with config label
            log_filename = f"train_logs_{config_label}_{agent_name}_seed{seed}.json"
            logger.save_logs(f"src/results/{log_filename}")
            
            dt = time.time() - t0
            print(f"    Training done in {dt:.1f}s (logs saved to {log_filename})")

            metrics = evaluate_agent(
                agent, env_config, num_episodes=num_eval, seed=base_seed + 5000
            )
            seed_results.append(metrics)
            print(
                f"    Cost: {metrics['mean_cost']:.1f} +/- {metrics['std_cost']:.1f}  "
                f"Service: {metrics['mean_service_level']:.3f}  "
                f"Stockout rate: {metrics['stockout_rate']:.3f}"
            )

        aggregated = {}
        for key in seed_results[0]:
            values = [r[key] for r in seed_results]
            aggregated[key] = float(np.mean(values))
            aggregated[f"{key}_across_seeds"] = float(np.std(values))
        results[agent_name] = aggregated

    return results


def run_delay_sweep(config: dict):
    """Sweep over observation delay levels."""
    env_base = config["environment"]
    agents_config = config["agents"]
    exp_config = config["experiment"]
    delay_levels = exp_config.get("delay_levels", [0, 1, 2, 4, 8])

    all_results = {}
    for delay in delay_levels:
        label = f"delay_{delay}"
        print(f"\n{'='*60}")
        print(f"Running delay sweep: observation_delay={delay}")
        print(f"{'='*60}")
        env_config = make_env_config(env_base, observation_delay=delay)
        all_results[label] = run_single_config(
            env_config, agents_config, exp_config, label
        )

    save_results({"experiment": "delay_sweep", "results": all_results}, "delay_sweep.json")
    return all_results


def run_noise_sweep(config: dict):
    """Sweep over noise standard deviation levels."""
    env_base = config["environment"]
    agents_config = config["agents"]
    exp_config = config["experiment"]
    noise_levels = exp_config.get("noise_levels", [0.0, 1.0, 3.0, 5.0, 10.0])

    all_results = {}
    for noise in noise_levels:
        label = f"noise_{noise}"
        print(f"\n{'='*60}")
        print(f"Running noise sweep: noise_std={noise}")
        print(f"{'='*60}")
        env_config = make_env_config(env_base, noise_std=noise)
        all_results[label] = run_single_config(
            env_config, agents_config, exp_config, label
        )

    save_results({"experiment": "noise_sweep", "results": all_results}, "noise_sweep.json")
    return all_results


def run_censoring_comparison(config: dict):
    """Compare censored vs uncensored demand feedback."""
    env_base = config["environment"]
    agents_config = config["agents"]
    exp_config = config["experiment"]

    all_results = {}
    for censor in [False, True]:
        label = f"censoring_{censor}"
        print(f"\n{'='*60}")
        print(f"Running censoring comparison: censoring={censor}")
        print(f"{'='*60}")
        env_config = make_env_config(env_base, censoring=censor)
        all_results[label] = run_single_config(
            env_config, agents_config, exp_config, label
        )

    save_results(
        {"experiment": "censoring_comparison", "results": all_results},
        "censoring_comparison.json",
    )
    return all_results


def run_full_benchmark(config: dict):
    """Run all benchmark experiments."""
    print("=" * 60)
    print("INVENTORY RL BENCHMARK")
    print("Comparing POMDP-aware (RecurrentPPO) vs conventional RL vs baselines")
    print("=" * 60)

    t0 = time.time()

    print("\n\n>>> EXPERIMENT 1: Delay Sweep")
    run_delay_sweep(config)

    print("\n\n>>> EXPERIMENT 2: Noise Sweep")
    run_noise_sweep(config)

    print("\n\n>>> EXPERIMENT 3: Censoring Comparison")
    run_censoring_comparison(config)

    total = time.time() - t0
    print(f"\n\nAll benchmarks completed in {total:.1f}s")
    print(f"Results saved in src/results/")


def main():
    parser = argparse.ArgumentParser(description="Run inventory RL benchmarks")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--experiment",
        choices=["all", "delay", "noise", "censoring"],
        default="all",
        help="Which experiment to run",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.experiment == "all":
        run_full_benchmark(config)
    elif args.experiment == "delay":
        run_delay_sweep(config)
    elif args.experiment == "noise":
        run_noise_sweep(config)
    elif args.experiment == "censoring":
        run_censoring_comparison(config)


if __name__ == "__main__":
    main()
