# Run: python -m src.visualize
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path(__file__).parent / "results"

AGENT_LABELS = {
    "ss_policy": "(s,S) Policy",
    "mpc": "MPC",
    "dqn": "DQN",
    "ppo": "PPO",
    "recurrent_ppo": "RecurrentPPO (LSTM)",
}

AGENT_COLORS = {
    "ss_policy": "#7f8c8d",
    "mpc": "#2c3e50",
    "dqn": "#e74c3c",
    "ppo": "#3498db",
    "recurrent_ppo": "#27ae60",
}

AGENT_ORDER = ["ss_policy", "mpc", "dqn", "ppo", "recurrent_ppo"]


def load_results(filename: str) -> dict:
    filepath = RESULTS_DIR / filename
    with open(filepath) as f:
        return json.load(f)


def plot_delay_sweep():
    """Plot total cost vs observation delay for all agents."""
    data = load_results("delay_sweep.json")
    results = data["results"]

    delays = sorted([int(k.split("_")[1]) for k in results.keys()])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for agent in AGENT_ORDER:
        costs = [results[f"delay_{d}"][agent]["mean_cost"] for d in delays]
        cost_stds = [results[f"delay_{d}"][agent]["std_cost"] for d in delays]
        axes[0].plot(
            delays,
            costs,
            "o-",
            label=AGENT_LABELS[agent],
            color=AGENT_COLORS[agent],
            linewidth=2,
            markersize=6,
        )
        axes[0].fill_between(
            delays,
            np.array(costs) - np.array(cost_stds),
            np.array(costs) + np.array(cost_stds),
            alpha=0.15,
            color=AGENT_COLORS[agent],
        )

    axes[0].set_xlabel("Observation Delay (steps)", fontsize=12)
    axes[0].set_ylabel("Mean Total Cost", fontsize=12)
    axes[0].set_title("Cost vs Observation Delay", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for agent in AGENT_ORDER:
        service = [
            results[f"delay_{d}"][agent]["mean_service_level"] for d in delays
        ]
        axes[1].plot(
            delays,
            service,
            "o-",
            label=AGENT_LABELS[agent],
            color=AGENT_COLORS[agent],
            linewidth=2,
            markersize=6,
        )
    axes[1].set_xlabel("Observation Delay (steps)", fontsize=12)
    axes[1].set_ylabel("Service Level", fontsize=12)
    axes[1].set_title("Service Level vs Observation Delay", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    for agent in AGENT_ORDER:
        stockout = [
            results[f"delay_{d}"][agent]["stockout_rate"] for d in delays
        ]
        axes[2].plot(
            delays,
            stockout,
            "o-",
            label=AGENT_LABELS[agent],
            color=AGENT_COLORS[agent],
            linewidth=2,
            markersize=6,
        )
    axes[2].set_xlabel("Observation Delay (steps)", fontsize=12)
    axes[2].set_ylabel("Stockout Rate", fontsize=12)
    axes[2].set_title("Stockout Rate vs Observation Delay", fontsize=13)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "delay_sweep.png", dpi=150, bbox_inches="tight")
    print("Saved delay_sweep.png")
    plt.close(fig)


def plot_noise_sweep():
    """Plot total cost vs noise level for all agents."""
    data = load_results("noise_sweep.json")
    results = data["results"]

    noise_levels = sorted([float(k.split("_")[1]) for k in results.keys()])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for agent in AGENT_ORDER:
        costs = [results[f"noise_{n}"][agent]["mean_cost"] for n in noise_levels]
        cost_stds = [results[f"noise_{n}"][agent]["std_cost"] for n in noise_levels]
        axes[0].plot(
            noise_levels,
            costs,
            "o-",
            label=AGENT_LABELS[agent],
            color=AGENT_COLORS[agent],
            linewidth=2,
            markersize=6,
        )
        axes[0].fill_between(
            noise_levels,
            np.array(costs) - np.array(cost_stds),
            np.array(costs) + np.array(cost_stds),
            alpha=0.15,
            color=AGENT_COLORS[agent],
        )

    axes[0].set_xlabel("Noise Std Dev", fontsize=12)
    axes[0].set_ylabel("Mean Total Cost", fontsize=12)
    axes[0].set_title("Cost vs Observation Noise", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for agent in AGENT_ORDER:
        service = [
            results[f"noise_{n}"][agent]["mean_service_level"] for n in noise_levels
        ]
        axes[1].plot(
            noise_levels,
            service,
            "o-",
            label=AGENT_LABELS[agent],
            color=AGENT_COLORS[agent],
            linewidth=2,
            markersize=6,
        )
    axes[1].set_xlabel("Noise Std Dev", fontsize=12)
    axes[1].set_ylabel("Service Level", fontsize=12)
    axes[1].set_title("Service Level vs Observation Noise", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    for agent in AGENT_ORDER:
        stockout = [
            results[f"noise_{n}"][agent]["stockout_rate"] for n in noise_levels
        ]
        axes[2].plot(
            noise_levels,
            stockout,
            "o-",
            label=AGENT_LABELS[agent],
            color=AGENT_COLORS[agent],
            linewidth=2,
            markersize=6,
        )
    axes[2].set_xlabel("Noise Std Dev", fontsize=12)
    axes[2].set_ylabel("Stockout Rate", fontsize=12)
    axes[2].set_title("Stockout Rate vs Observation Noise", fontsize=13)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "noise_sweep.png", dpi=150, bbox_inches="tight")
    print("Saved noise_sweep.png")
    plt.close(fig)


def plot_censoring_comparison():
    """Bar chart comparing agents with and without demand censoring."""
    data = load_results("censoring_comparison.json")
    results = data["results"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(AGENT_ORDER))
    width = 0.35

    for idx, (metric, ylabel, title) in enumerate(
        [
            ("mean_cost", "Mean Total Cost", "Cost: Censored vs Uncensored"),
            ("mean_service_level", "Service Level", "Service Level: Censored vs Uncensored"),
            ("stockout_rate", "Stockout Rate", "Stockout Rate: Censored vs Uncensored"),
        ]
    ):
        uncensored = [results["censoring_False"][a][metric] for a in AGENT_ORDER]
        censored = [results["censoring_True"][a][metric] for a in AGENT_ORDER]

        bars1 = axes[idx].bar(
            x - width / 2,
            uncensored,
            width,
            label="Uncensored",
            color="#3498db",
            alpha=0.8,
        )
        bars2 = axes[idx].bar(
            x + width / 2,
            censored,
            width,
            label="Censored",
            color="#e74c3c",
            alpha=0.8,
        )

        axes[idx].set_xlabel("Agent", fontsize=11)
        axes[idx].set_ylabel(ylabel, fontsize=11)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(
            [AGENT_LABELS[a] for a in AGENT_ORDER], rotation=25, ha="right", fontsize=9
        )
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(
        RESULTS_DIR / "censoring_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("Saved censoring_comparison.png")
    plt.close(fig)


def plot_agent_comparison_bar():
    """Bar chart of all agents on default config (delay=2, noise=3, censoring=True)."""
    data = load_results("delay_sweep.json")
    results = data["results"]
    default_key = "delay_2"

    if default_key not in results:
        keys = list(results.keys())
        default_key = keys[len(keys) // 2]

    agents_data = results[default_key]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(AGENT_ORDER))

    colors = [AGENT_COLORS[a] for a in AGENT_ORDER]
    labels = [AGENT_LABELS[a] for a in AGENT_ORDER]

    costs = [agents_data[a]["mean_cost"] for a in AGENT_ORDER]
    cost_stds = [agents_data[a]["std_cost"] for a in AGENT_ORDER]
    axes[0].bar(x, costs, color=colors, alpha=0.85, yerr=cost_stds, capsize=4)
    axes[0].set_ylabel("Mean Total Cost", fontsize=11)
    axes[0].set_title("Total Cost Comparison (delay=2, noise=3)", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    service = [agents_data[a]["mean_service_level"] for a in AGENT_ORDER]
    axes[1].bar(x, service, color=colors, alpha=0.85)
    axes[1].set_ylabel("Service Level", fontsize=11)
    axes[1].set_title("Service Level Comparison", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3, axis="y")

    stockout = [agents_data[a]["stockout_rate"] for a in AGENT_ORDER]
    axes[2].bar(x, stockout, color=colors, alpha=0.85)
    axes[2].set_ylabel("Stockout Rate", fontsize=11)
    axes[2].set_title("Stockout Rate Comparison", fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(
        RESULTS_DIR / "agent_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("Saved agent_comparison.png")
    plt.close(fig)


def plot_robustness_heatmap():
    """Heatmap showing cost degradation across delay x noise conditions."""
    delay_data = load_results("delay_sweep.json")["results"]
    noise_data = load_results("noise_sweep.json")["results"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    delays = sorted([int(k.split("_")[1]) for k in delay_data.keys()])
    agents = AGENT_ORDER

    delay_matrix = np.array(
        [[delay_data[f"delay_{d}"][a]["mean_cost"] for d in delays] for a in agents]
    )

    sns.heatmap(
        delay_matrix,
        ax=axes[0],
        xticklabels=delays,
        yticklabels=[AGENT_LABELS[a] for a in agents],
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Cost"},
    )
    axes[0].set_xlabel("Observation Delay", fontsize=11)
    axes[0].set_title("Cost Heatmap: Delay Sweep", fontsize=12)

    noise_levels = sorted([float(k.split("_")[1]) for k in noise_data.keys()])

    noise_matrix = np.array(
        [
            [noise_data[f"noise_{n}"][a]["mean_cost"] for n in noise_levels]
            for a in agents
        ]
    )

    sns.heatmap(
        noise_matrix,
        ax=axes[1],
        xticklabels=[f"{n:.0f}" for n in noise_levels],
        yticklabels=[AGENT_LABELS[a] for a in agents],
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Cost"},
    )
    axes[1].set_xlabel("Noise Std Dev", fontsize=11)
    axes[1].set_title("Cost Heatmap: Noise Sweep", fontsize=12)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "robustness_heatmap.png", dpi=150, bbox_inches="tight")
    print("Saved robustness_heatmap.png")
    plt.close(fig)


def plot_pomdp_advantage():
    """Plot showing the advantage of POMDP-aware RL over conventional methods.

    Shows the percentage cost improvement of RecurrentPPO over each other agent
    as delay and noise increase.
    """
    delay_data = load_results("delay_sweep.json")["results"]
    noise_data = load_results("noise_sweep.json")["results"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    delays = sorted([int(k.split("_")[1]) for k in delay_data.keys()])
    for agent in ["ss_policy", "mpc", "dqn", "ppo"]:
        improvements = []
        for d in delays:
            rppo_cost = delay_data[f"delay_{d}"]["recurrent_ppo"]["mean_cost"]
            agent_cost = delay_data[f"delay_{d}"][agent]["mean_cost"]
            improvement = (agent_cost - rppo_cost) / agent_cost * 100
            improvements.append(improvement)
        axes[0].plot(
            delays,
            improvements,
            "o-",
            label=f"vs {AGENT_LABELS[agent]}",
            color=AGENT_COLORS[agent],
            linewidth=2,
        )

    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[0].set_xlabel("Observation Delay (steps)", fontsize=12)
    axes[0].set_ylabel("Cost Improvement (%)", fontsize=12)
    axes[0].set_title("RecurrentPPO Advantage vs Delay", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    noise_levels = sorted([float(k.split("_")[1]) for k in noise_data.keys()])
    for agent in ["ss_policy", "mpc", "dqn", "ppo"]:
        improvements = []
        for n in noise_levels:
            rppo_cost = noise_data[f"noise_{n}"]["recurrent_ppo"]["mean_cost"]
            agent_cost = noise_data[f"noise_{n}"][agent]["mean_cost"]
            improvement = (agent_cost - rppo_cost) / max(agent_cost, 1e-8) * 100
            improvements.append(improvement)
        axes[1].plot(
            noise_levels,
            improvements,
            "o-",
            label=f"vs {AGENT_LABELS[agent]}",
            color=AGENT_COLORS[agent],
            linewidth=2,
        )

    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[1].set_xlabel("Noise Std Dev", fontsize=12)
    axes[1].set_ylabel("Cost Improvement (%)", fontsize=12)
    axes[1].set_title("RecurrentPPO Advantage vs Noise", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "pomdp_advantage.png", dpi=150, bbox_inches="tight")
    print("Saved pomdp_advantage.png")
    plt.close(fig)


def plot_summary_table():
    """Generate a text-based summary table of all results."""
    delay_data = load_results("delay_sweep.json")["results"]

    default_key = "delay_2"
    if default_key not in delay_data:
        keys = list(delay_data.keys())
        default_key = keys[len(keys) // 2]

    print("\n" + "=" * 80)
    print("SUMMARY: Agent Performance (delay=2, noise=3, censoring=True)")
    print("=" * 80)
    print(f"{'Agent':<25} {'Cost':>10} {'Service':>10} {'Stockout':>10}")
    print("-" * 55)

    for agent in AGENT_ORDER:
        m = delay_data[default_key][agent]
        print(
            f"{AGENT_LABELS[agent]:<25} "
            f"{m['mean_cost']:>10.1f} "
            f"{m['mean_service_level']:>10.4f} "
            f"{m['stockout_rate']:>10.4f}"
        )


def generate_all_plots():
    """Generate all visualization plots from saved results."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11})

    available = []
    for f in ["delay_sweep.json", "noise_sweep.json", "censoring_comparison.json"]:
        if (RESULTS_DIR / f).exists():
            available.append(f)
        else:
            print(f"Warning: {f} not found, skipping related plots")

    if "delay_sweep.json" in available:
        plot_delay_sweep()
        plot_agent_comparison_bar()
        plot_summary_table()

    if "noise_sweep.json" in available:
        plot_noise_sweep()

    if "censoring_comparison.json" in available:
        plot_censoring_comparison()

    if "delay_sweep.json" in available and "noise_sweep.json" in available:
        plot_robustness_heatmap()
        plot_pomdp_advantage()

    print(f"\nAll plots saved to {RESULTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument(
        "--plot",
        choices=[
            "all",
            "delay",
            "noise",
            "censoring",
            "comparison",
            "heatmap",
            "advantage",
        ],
        default="all",
        help="Which plot to generate",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", palette="muted")

    if args.plot == "all":
        generate_all_plots()
    elif args.plot == "delay":
        plot_delay_sweep()
    elif args.plot == "noise":
        plot_noise_sweep()
    elif args.plot == "censoring":
        plot_censoring_comparison()
    elif args.plot == "comparison":
        plot_agent_comparison_bar()
    elif args.plot == "heatmap":
        plot_robustness_heatmap()
    elif args.plot == "advantage":
        plot_pomdp_advantage()


if __name__ == "__main__":
    main()
