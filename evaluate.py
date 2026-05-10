"""
evaluate.py -- Load saved policies and produce detailed evaluation reports
Usage:
    python evaluate.py --policy policies/policy_v2.pkl --config configs/qlearning_v1.yaml
    python evaluate.py --policy policies/policy_v1.pkl --compare policies/policy_v2.pkl
"""

import argparse
import os
import sys
from pathlib import Path

# Change working directory to script location to resolve relative paths
os.chdir(Path(__file__).parent)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from sim.environment import (
    StreetLightEnv, state_to_index, STATE_SIZE, NUM_ACTIONS,
    BRIGHTNESS_LEVELS, BRIGHTNESS_WATTS,
)
from train import QLearningAgent, fixed_timer_action, PALETTE, _style_ax


# ============================================================================
# Evaluation runner
# ============================================================================

def evaluate_policy(agent: QLearningAgent | None, cfg: dict,
                    label: str = "RL", seed_offset: int = 0) -> dict:
    """
    Run `episodes` full episodes and collect per-hour statistics.
    agent=None -> fixed timer baseline.
    """
    env_cfg  = cfg["environment"]
    eval_cfg = cfg["evaluation"]
    seed     = cfg["run"].get("seed", 42) + seed_offset

    env = StreetLightEnv(episode_length=env_cfg["episode_length"], seed=seed)
    episodes = eval_cfg["episodes"]

    all_rewards     = []
    all_energies    = []
    all_unnecessary = []
    action_counts   = np.zeros(NUM_ACTIONS, dtype=int)
    hourly_actions  = {h: np.zeros(NUM_ACTIONS, dtype=int) for h in range(24)}

    if agent is not None:
        agent.epsilon = 0.0   # greedy

    for _ in range(episodes):
        state     = env.reset()
        state_idx = state_to_index(state)
        ep_reward = 0.0
        done      = False

        while not done:
            hour = state[0]
            if agent is None:
                action = fixed_timer_action(hour)
            else:
                action = agent.choose_action(state_idx)

            step       = env.step(action)
            ep_reward += step.reward
            action_counts[action] += 1
            hourly_actions[hour][action] += 1

            state     = step.state
            state_idx = state_to_index(state)
            done      = step.done

        all_rewards.append(ep_reward)
        all_energies.append(env.energy_used)
        all_unnecessary.append(env.unnecessary_lighting_pct)

    return {
        "label":              label,
        "avg_reward":         np.mean(all_rewards),
        "std_reward":         np.std(all_rewards),
        "avg_energy":         np.mean(all_energies),
        "std_energy":         np.std(all_energies),
        "avg_unnecessary":    np.mean(all_unnecessary),
        "action_counts":      action_counts,
        "hourly_actions":     hourly_actions,
        "all_rewards":        all_rewards,
        "all_energies":       all_energies,
    }


# ============================================================================
# Plots
# ============================================================================

def plot_action_distribution(results: list, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Action Distribution per Policy",
                 color=PALETTE["text"], fontsize=12, fontweight="bold")

    action_colors = ["#546E7A", "#4FC3F7", "#FFA726", "#EF5350"]

    for ax, res in zip(axes, results):
        _style_ax(ax, res["label"])
        counts = res["action_counts"]
        total  = counts.sum()
        pcts   = 100.0 * counts / total
        bars   = ax.bar(list(BRIGHTNESS_LEVELS.values()), pcts,
                        color=action_colors, edgecolor=PALETTE["bg"], linewidth=1.5)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{pct:.1f}%", ha="center", color=PALETTE["text"], fontsize=9)
        ax.set_ylabel("% of Steps")
        ax.set_ylim(0, 100)

    plt.tight_layout()
    path = os.path.join(plot_dir, "action_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [CHART] Action distribution saved -> {path}")


def plot_hourly_heatmap(results: list, plot_dir: str):
    """Heatmap of most-chosen action per hour for each policy."""
    os.makedirs(plot_dir, exist_ok=True)
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Preferred Action per Hour (Heatmap)",
                 color=PALETTE["text"], fontsize=12, fontweight="bold")

    cmap = plt.get_cmap("RdYlGn", 4)
    action_names = list(BRIGHTNESS_LEVELS.values())

    for ax, res in zip(axes, results):
        _style_ax(ax, res["label"])
        matrix = np.array([res["hourly_actions"][h] for h in range(24)]).T  # shape (4, 24)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                       vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax.set_yticks(range(4))
        ax.set_yticklabels(action_names, color=PALETTE["text"])
        ax.set_xlabel("Hour of Day", color=PALETTE["text"])
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], color=PALETTE["text"])
        cbar = fig.colorbar(im, ax=ax, ticks=range(4))
        cbar.ax.set_yticklabels(action_names, color=PALETTE["text"])
        cbar.ax.tick_params(colors=PALETTE["text"])

    plt.tight_layout()
    path = os.path.join(plot_dir, "hourly_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [MAP]  Hourly heatmap saved -> {path}")


def plot_reward_box(results: list, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Reward & Energy Distribution across Evaluation Episodes",
                 color=PALETTE["text"], fontsize=12, fontweight="bold")

    colors_list = [PALETTE["rl"], PALETTE["baseline"], "#CE93D8"][:len(results)]

    for ax, key, title, unit in [
        (axes[0], "all_rewards",  "Episode Reward",     "Reward"),
        (axes[1], "all_energies", "Energy Used",        "Wh"),
    ]:
        _style_ax(ax, title)
        data   = [r[key] for r in results]
        labels = [r["label"] for r in results]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color=PALETTE["accent"], linewidth=2),
                        whiskerprops=dict(color=PALETTE["text"]),
                        capprops=dict(color=PALETTE["text"]),
                        flierprops=dict(markerfacecolor=PALETTE["text"], markersize=3))
        for patch, c in zip(bp["boxes"], colors_list):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_ylabel(unit, color=PALETTE["text"])
        ax.tick_params(colors=PALETTE["text"])

    plt.tight_layout()
    path = os.path.join(plot_dir, "reward_energy_box.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [BOX] Box plot saved -> {path}")


# ============================================================================
# Pretty print summary
# ============================================================================

def print_eval_table(results: list):
    sep = "-" * 70
    print(f"\n  {sep}")
    print(f"  {'DETAILED EVALUATION REPORT':^68}")
    print(f"  {sep}")
    header = f"  {'Metric':<30}"
    for r in results:
        header += f"  {r['label']:>12}"
    print(header)
    print(f"  {sep}")

    metrics = [
        ("Avg Episode Reward",     "avg_reward",      ".1f"),
        ("Std Reward",             "std_reward",       ".1f"),
        ("Avg Energy (Wh/day)",    "avg_energy",       ".0f"),
        ("Std Energy",             "std_energy",       ".0f"),
        ("Unnecessary Light (%)",  "avg_unnecessary",  ".1f"),
    ]
    for name, key, fmt in metrics:
        row = f"  {name:<30}"
        for r in results:
            row += f"  {r[key]:>12{fmt}}"
        print(row)

    # action breakdown
    print(f"\n  {'Action Usage (% of steps)':<30}")
    for ai, aname in BRIGHTNESS_LEVELS.items():
        row = f"    {aname:<28}"
        for r in results:
            total = r["action_counts"].sum()
            pct   = 100.0 * r["action_counts"][ai] / total
            row  += f"  {pct:>11.1f}%"
        print(row)

    print(f"  {sep}\n")


# ============================================================================
# Entry point
# ============================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Street Lighting RL Policies")
    parser.add_argument("--policy",  type=str, default="policies/policy_v2.pkl",
                        help="Path to primary policy .pkl")
    parser.add_argument("--compare", type=str, default=None,
                        help="Optional second policy .pkl for comparison")
    parser.add_argument("--config",  type=str, default="configs/qlearning_v1.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    plot_dir = cfg["logging"]["plot_dir"]

    results = []

    # -- Load primary policy -----------------------------------------------
    print(f"\n  Loading policy: {args.policy}")
    agent_v2 = QLearningAgent.load(args.policy)
    label    = Path(args.policy).stem.replace("_", " ").title()
    results.append(evaluate_policy(agent_v2, cfg, label=label, seed_offset=200))

    # -- Optional second policy --------------------------------------------
    if args.compare:
        print(f"  Loading policy: {args.compare}")
        agent_v1 = QLearningAgent.load(args.compare)
        label2   = Path(args.compare).stem.replace("_", " ").title()
        results.append(evaluate_policy(agent_v1, cfg, label=label2, seed_offset=300))

    # -- Baseline ----------------------------------------------------------
    results.append(evaluate_policy(None, cfg, label="Fixed Timer", seed_offset=400))

    # -- Report & plots ----------------------------------------------------
    print_eval_table(results)
    plot_action_distribution(results, plot_dir)
    plot_hourly_heatmap(results, plot_dir)
    plot_reward_box(results, plot_dir)

    print("  [DONE] Evaluation complete. Plots saved to:", plot_dir, "\n")


if __name__ == "__main__":
    main()
