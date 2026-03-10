# Run: python -m src.experiments.run_single --agent recurrent_ppo
import argparse
import time

from src.agents import (
    DQNAgent,
    QRDQNAgent,
    A2CAgent,
    MPCAgent,
    PPOAgent,
    RecurrentPPOAgent,
    SSPolicy,
)
from src.environment import InventoryEnv
from src.utils import evaluate_agent, load_config, save_results, set_seed

AGENT_MAP = {
    "ss_policy": SSPolicy,
    "mpc": MPCAgent,
    "dqn": DQNAgent,
    "qrdqn": QRDQNAgent,
    "a2c": A2CAgent,
    "ppo": PPOAgent,
    "recurrent_ppo": RecurrentPPOAgent,
}


def main():
    parser = argparse.ArgumentParser(description="Run a single inventory RL agent")
    parser.add_argument(
        "--agent",
        required=True,
        choices=list(AGENT_MAP.keys()),
        help="Agent to run",
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Training timesteps (RL agents)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=50, help="Number of evaluation episodes"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    env_config = config["environment"]
    agent_config = config["agents"][args.agent]

    set_seed(args.seed)

    print(f"Agent: {args.agent}")
    print(f"Env config: delay={env_config['observation_delay']}, "
          f"noise={env_config['noise_std']}, censoring={env_config['censoring']}")

    agent_cls = AGENT_MAP[args.agent]
    agent = agent_cls(agent_config, env_config)

    if args.agent in {"dqn", "qrdqn", "a2c", "ppo", "recurrent_ppo"}:
        timesteps = args.timesteps or agent_config.get("total_timesteps", 500000)
        print(f"Training for {timesteps} timesteps...")
        t0 = time.time()
        env = InventoryEnv(env_config)
        agent.train(env, total_timesteps=timesteps, seed=args.seed)
        print(f"Training completed in {time.time() - t0:.1f}s")

    print(f"\nEvaluating over {args.eval_episodes} episodes...")
    metrics = evaluate_agent(
        agent, env_config, num_episodes=args.eval_episodes, seed=args.seed + 5000
    )

    print(f"\nResults:")
    print(f"  Mean cost:      {metrics['mean_cost']:.2f} +/- {metrics['std_cost']:.2f}")
    print(f"  Mean reward:    {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"  Service level:  {metrics['mean_service_level']:.4f}")
    print(f"  Stockout rate:  {metrics['stockout_rate']:.4f}")

    save_results(
        {"agent": args.agent, "seed": args.seed, "metrics": metrics},
        f"single_{args.agent}_seed{args.seed}_{timesteps}.json",
    )


if __name__ == "__main__":
    main()
