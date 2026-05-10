"""
train.py – Main training script for Smart Adaptive Street Lighting RL
Usage:
    python train.py --config configs/qlearning_v1.yaml
    python train.py --config configs/qlearning_v1.yaml --episodes 2000
"""

import argparse
import csv
import os
import pickle
import random
import sys
import time
from pathlib import Path

# Change working directory to script location to resolve relative paths
os.chdir(Path(__file__).parent)

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")          # headless backend – safe for all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mlflow
import mlflow.sklearn

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from sim.environment import (
    StreetLightEnv, state_to_index, STATE_SIZE, NUM_ACTIONS,
    BRIGHTNESS_WATTS, BRIGHTNESS_LEVELS,
)


# ════════════════════════════════════════════════════════════════════════════
# Q-Learning Agent
# ════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """Tabular Q-Learning with epsilon-greedy exploration."""

    def __init__(self,
                 state_size:      int,
                 action_size:     int,
                 learning_rate:   float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon:         float = 0.1,
                 epsilon_min:     float = 0.01,
                 epsilon_decay:   float = 0.995,
                 seed:            int   = 42):
        self.state_size      = state_size
        self.action_size     = action_size
        self.lr              = learning_rate
        self.gamma           = discount_factor
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.rng             = random.Random(seed)
        np.random.seed(seed)

        # Q-table: rows = states, cols = actions
        self.Q = np.zeros((state_size, action_size), dtype=np.float32)

    # ------------------------------------------------------------------
    def choose_action(self, state_idx: int) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.action_size - 1)
        return int(np.argmax(self.Q[state_idx]))

    # ------------------------------------------------------------------
    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.lr * (target - self.Q[s, a])

    # ------------------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"Q": self.Q, "epsilon": self.epsilon,
                         "lr": self.lr, "gamma": self.gamma}, f)
        print(f"  [OK] Policy saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(state_size=STATE_SIZE, action_size=NUM_ACTIONS,
                    learning_rate=data["lr"], discount_factor=data["gamma"],
                    epsilon=0.0)
        agent.Q = data["Q"]
        return agent


# ════════════════════════════════════════════════════════════════════════════
# Baseline: Fixed-Timer Policy
# ════════════════════════════════════════════════════════════════════════════

def fixed_timer_action(hour: int) -> int:
    """Always full brightness between 18:00-06:00, off otherwise."""
    if hour >= 18 or hour < 6:
        return 3   # Full
    return 0       # Off


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

def train(cfg: dict, override_episodes: int | None = None) -> QLearningAgent:
    run_cfg   = cfg["run"]
    env_cfg   = cfg["environment"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["training"]
    log_cfg   = cfg["logging"]
    pol_cfg   = cfg["policy"]

    total_episodes = override_episodes or train_cfg["episodes"]
    seed           = run_cfg.get("seed", 42)

    print("\n" + "=" * 60)
    print("  Smart Adaptive Street Lighting -- Q-Learning Training")
    print("  SDG 7 (Clean Energy) + SDG 11 (Sustainable Cities)")
    print("=" * 60)
    print(f"  Run ID     : {run_cfg['id']}")
    print(f"  Episodes   : {total_episodes}")
    print(f"  Seed       : {seed}")
    print(f"  Algorithm  : {agent_cfg['algorithm'].upper()}")
    print(f"  lr (alpha) : {agent_cfg['learning_rate']}")
    print(f"  gamma      : {agent_cfg['discount_factor']}")
    print(f"  epsilon    : {agent_cfg['epsilon']}")
    print("=" * 60 + "\n")

    env = StreetLightEnv(episode_length=env_cfg["episode_length"], seed=seed)
    agent = QLearningAgent(
        state_size      = STATE_SIZE,
        action_size     = NUM_ACTIONS,
        learning_rate   = agent_cfg["learning_rate"],
        discount_factor = agent_cfg["discount_factor"],
        epsilon         = agent_cfg["epsilon"],
        epsilon_min     = agent_cfg["epsilon_min"],
        epsilon_decay   = agent_cfg["epsilon_decay"],
        seed            = seed,
    )

    # ── tracking buffers ─────────────────────────────────────────────────
    episode_rewards  = []
    episode_energies = []
    log_every        = train_cfg["log_every"]
    v1_saved         = False
    start_t          = time.time()

    for ep in range(1, total_episodes + 1):
        state     = env.reset()
        state_idx = state_to_index(state)
        ep_reward = 0.0
        done      = False

        while not done:
            action              = agent.choose_action(state_idx)
            step                = env.step(action)
            next_idx            = state_to_index(step.state)
            agent.update(state_idx, action, step.reward, next_idx, step.done)
            ep_reward          += step.reward
            state_idx           = next_idx
            done                = step.done

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        episode_energies.append(env.energy_used)

        # save policy_v1 at checkpoint
        if ep == pol_cfg["v1_checkpoint"] and not v1_saved:
            agent.save(os.path.join(pol_cfg["save_dir"], "policy_v1.pkl"))
            v1_saved = True

        # ── progress print ────────────────────────────────────────────
        if ep % log_every == 0:
            window  = min(log_every, len(episode_rewards))
            avg_r   = np.mean(episode_rewards[-window:])
            avg_e   = np.mean(episode_energies[-window:])
            elapsed = time.time() - start_t
            print(f"  Ep {ep:>5}/{total_episodes}  "
                  f"avg_reward={avg_r:+.1f}  "
                  f"avg_energy={avg_e:.0f}Wh  "
                  f"eps={agent.epsilon:.4f}  "
                  f"[{elapsed:.1f}s]")

    # save final policy_v2
    agent.save(os.path.join(pol_cfg["save_dir"], "policy_v2.pkl"))

    # also save v1 if checkpoint not reached
    if not v1_saved:
        agent.save(os.path.join(pol_cfg["save_dir"], "policy_v1.pkl"))

    print(f"\n  Training complete in {time.time()-start_t:.1f}s")
    return agent, episode_rewards, episode_energies


# ════════════════════════════════════════════════════════════════════════════
# Baseline evaluation helper
# ════════════════════════════════════════════════════════════════════════════

def run_baseline(cfg: dict) -> tuple:
    """Run fixed-timer policy for cfg[evaluation][episodes] episodes."""
    env_cfg  = cfg["environment"]
    eval_cfg = cfg["evaluation"]
    seed     = cfg["run"].get("seed", 42)

    env = StreetLightEnv(episode_length=env_cfg["episode_length"], seed=seed + 99)
    rewards, energies, unnecessary = [], [], []

    for _ in range(eval_cfg["episodes"]):
        state = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            hour   = state[0]
            action = fixed_timer_action(hour)
            step   = env.step(action)
            ep_reward += step.reward
            state      = step.state
            done       = step.done
        rewards.append(ep_reward)
        energies.append(env.energy_used)
        unnecessary.append(env.unnecessary_lighting_pct)

    return (np.mean(rewards), np.mean(energies), np.mean(unnecessary))


def run_rl_eval(agent: QLearningAgent, cfg: dict) -> tuple:
    """Evaluate trained RL agent (greedy, no exploration)."""
    env_cfg  = cfg["environment"]
    eval_cfg = cfg["evaluation"]
    seed     = cfg["run"].get("seed", 42)

    env = StreetLightEnv(episode_length=env_cfg["episode_length"], seed=seed + 100)
    saved_eps   = agent.epsilon
    agent.epsilon = 0.0          # greedy evaluation

    rewards, energies, unnecessary = [], [], []

    for _ in range(eval_cfg["episodes"]):
        state = env.reset()
        state_idx = state_to_index(state)
        ep_reward, done = 0.0, False
        while not done:
            action    = agent.choose_action(state_idx)
            step      = env.step(action)
            ep_reward += step.reward
            state_idx  = state_to_index(step.state)
            done       = step.done
        rewards.append(ep_reward)
        energies.append(env.energy_used)
        unnecessary.append(env.unnecessary_lighting_pct)

    agent.epsilon = saved_eps
    return (np.mean(rewards), np.mean(energies), np.mean(unnecessary))


# ════════════════════════════════════════════════════════════════════════════
# CSV logging
# ════════════════════════════════════════════════════════════════════════════

def log_results(cfg: dict,
                agent: QLearningAgent,
                episode_rewards: list,
                episode_energies: list,
                rl_avg_reward: float,
                rl_avg_energy: float,
                baseline_avg_reward: float,
                baseline_avg_energy: float):
    log_cfg   = cfg["logging"]
    train_cfg = cfg["training"]
    agent_cfg = cfg["agent"]
    run_cfg   = cfg["run"]
    cost_kwh  = cfg["energy"]["cost_per_kwh"]

    os.makedirs(os.path.dirname(log_cfg["results_file"]), exist_ok=True)
    file_exists = os.path.isfile(log_cfg["results_file"])

    energy_saved_pct = 100.0 * (baseline_avg_energy - rl_avg_energy) / (baseline_avg_energy + 1e-9)
    cost_saved       = (baseline_avg_energy - rl_avg_energy) / 1000.0 * cost_kwh   # USD per day

    row = {
        "run_id":               run_cfg["id"],
        "episodes":             train_cfg["episodes"],
        "avg_reward_rl":        round(rl_avg_reward, 4),
        "avg_reward_baseline":  round(baseline_avg_reward, 4),
        "avg_energy_rl_wh":     round(rl_avg_energy, 2),
        "avg_energy_baseline_wh": round(baseline_avg_energy, 2),
        "energy_saved_pct":     round(energy_saved_pct, 2),
        "cost_saved_usd_day":   round(cost_saved, 4),
        "epsilon":              round(agent.epsilon, 6),
        "learning_rate":        agent_cfg["learning_rate"],
        "discount_factor":      agent_cfg["discount_factor"],
        "description":          run_cfg.get("description", ""),
    }

    with open(log_cfg["results_file"], "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # MLFlow Tracking
    mlflow.set_experiment("Smart_Street_Lighting_RL")
    with mlflow.start_run(run_name=run_cfg["id"]):
        # Log parameters
        mlflow.log_params({
            "episodes": train_cfg["episodes"],
            "learning_rate": agent_cfg["learning_rate"],
            "discount_factor": agent_cfg["discount_factor"],
            "epsilon_start": agent_cfg["epsilon"],
            "epsilon_decay": agent_cfg["epsilon_decay"],
            "description": run_cfg.get("description", "")
        })
        
        # Log metrics
        mlflow.log_metrics({
            "avg_reward_rl": round(rl_avg_reward, 4),
            "avg_reward_baseline": round(baseline_avg_reward, 4),
            "avg_energy_rl_wh": round(rl_avg_energy, 2),
            "avg_energy_baseline_wh": round(baseline_avg_energy, 2),
            "energy_saved_pct": round(energy_saved_pct, 2),
            "cost_saved_usd_day": round(cost_saved, 4)
        })
        
        # Log the trained model
        mlflow.log_artifact(os.path.join(cfg["policy"]["save_dir"], "policy_v2.pkl"), "policies")

    print(f"\n  [LOG] Results logged to CSV -> {log_cfg['results_file']}")
    print(f"  [LOG] Results also tracked in MLflow (Experiment: Smart_Street_Lighting_RL)")
    return row


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "rl":       "#4FC3F7",
    "baseline": "#EF9A9A",
    "accent":   "#A5D6A7",
    "bg":       "#0D1117",
    "surface":  "#161B22",
    "text":     "#E6EDF3",
    "grid":     "#21262D",
}

def _style_ax(ax, title: str = ""):
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=11, fontweight="bold", pad=8)


def plot_reward_curve(episode_rewards: list, plot_dir: str):
    """Smoothed average reward over training episodes."""
    os.makedirs(plot_dir, exist_ok=True)
    rewards = np.array(episode_rewards)
    window  = max(1, len(rewards) // 50)
    smooth  = np.convolve(rewards, np.ones(window) / window, mode="valid")
    x_raw   = np.arange(1, len(rewards) + 1)
    x_sm    = np.arange(window, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    _style_ax(ax, "Average Reward Over Training Episodes")

    ax.plot(x_raw, rewards, color=PALETTE["rl"], alpha=0.2, linewidth=0.7, label="Raw reward")
    ax.plot(x_sm,  smooth,  color=PALETTE["rl"], linewidth=2.2, label=f"Smoothed (w={window})")
    ax.axhline(0, color=PALETTE["grid"], linewidth=0.8, linestyle=":")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"], framealpha=0.8)

    # annotation
    max_idx = int(np.argmax(smooth))
    ax.annotate(f"Peak: {smooth[max_idx]:.1f}",
                xy=(x_sm[max_idx], smooth[max_idx]),
                xytext=(x_sm[max_idx] + len(smooth)*0.05, smooth[max_idx]),
                color=PALETTE["accent"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))

    plt.tight_layout()
    path = os.path.join(plot_dir, "reward_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Reward curve saved -> {path}")


def plot_energy_comparison(rl_energy: float, baseline_energy: float,
                           rl_unnecessary: float, baseline_unnecessary: float,
                           plot_dir: str, cost_kwh: float = 0.12):
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "RL Policy vs Fixed-Timer Baseline – Energy & Cost Comparison\n"
        "SDG 7 (Clean Energy) · SDG 11 (Sustainable Cities)",
        color=PALETTE["text"], fontsize=12, fontweight="bold", y=1.02,
    )

    labels = ["RL Policy", "Fixed Timer"]
    colors = [PALETTE["rl"], PALETTE["baseline"]]

    # ── Panel 1: Energy used ───────────────────────────────────────────
    ax = axes[0]
    _style_ax(ax, "Avg Energy Used per Day (Wh)")
    bars = ax.bar(labels, [rl_energy, baseline_energy], color=colors, width=0.5,
                  edgecolor=PALETTE["bg"], linewidth=1.5)
    for bar, val in zip(bars, [rl_energy, baseline_energy]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{val:.0f} Wh", ha="center", color=PALETTE["text"], fontsize=9)
    ax.set_ylabel("Watt-hours")

    # ── Panel 2: Unnecessary lighting % ───────────────────────────────
    ax = axes[1]
    _style_ax(ax, "Unnecessary Lighting (% of Steps)")
    bars = ax.bar(labels, [rl_unnecessary, baseline_unnecessary], color=colors,
                  width=0.5, edgecolor=PALETTE["bg"], linewidth=1.5)
    for bar, val in zip(bars, [rl_unnecessary, baseline_unnecessary]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", color=PALETTE["text"], fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("% Steps with Unnecessary Full Brightness")

    # ── Panel 3: Cost saved ────────────────────────────────────────────
    ax = axes[2]
    _style_ax(ax, "Estimated Cost Saved (USD/day)")
    cost_rl       = rl_energy       / 1000.0 * cost_kwh
    cost_baseline = baseline_energy / 1000.0 * cost_kwh
    saved         = max(0, cost_baseline - cost_rl)
    bars = ax.bar(["Baseline Cost", "RL Cost", "Saved"],
                  [cost_baseline, cost_rl, saved],
                  color=[PALETTE["baseline"], PALETTE["rl"], PALETTE["accent"]],
                  width=0.5, edgecolor=PALETTE["bg"], linewidth=1.5)
    for bar, val in zip(bars, [cost_baseline, cost_rl, saved]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"${val:.4f}", ha="center", color=PALETTE["text"], fontsize=8)
    ax.set_ylabel("USD per day")

    plt.tight_layout()
    path = os.path.join(plot_dir, "energy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Energy comparison chart saved -> {path}")


# ============================================================================
# Print comparison summary table
# ============================================================================

def print_comparison(row: dict, rl_unnecessary: float, baseline_unnecessary: float):
    sep = "-" * 55
    print(f"\n  {sep}")
    print(f"  {'POLICY COMPARISON SUMMARY':^53}")
    print(f"  {sep}")
    print(f"  {'Metric':<35} {'RL':>8}  {'Baseline':>8}")
    print(f"  {sep}")
    print(f"  {'Avg Episode Reward':<35} {row['avg_reward_rl']:>8.1f}  {row['avg_reward_baseline']:>8.1f}")
    print(f"  {'Avg Energy Used (Wh/day)':<35} {row['avg_energy_rl_wh']:>8.0f}  {row['avg_energy_baseline_wh']:>8.0f}")
    print(f"  {'Unnecessary Lighting (%)':<35} {rl_unnecessary:>7.1f}%  {baseline_unnecessary:>7.1f}%")
    print(f"  {'Energy Saved (%)':<35} {row['energy_saved_pct']:>7.2f}%  {'--':>8}")
    print(f"  {'Cost Saved (USD/day)':<35} ${row['cost_saved_usd_day']:>7.4f}  {'--':>8}")
    print(f"  {sep}\n")


# ============================================================================
# Entry point
# ============================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Street Lighting RL Agent")
    parser.add_argument("--config",   type=str, default="configs/qlearning_v1.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of training episodes")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Training ──────────────────────────────────────────────────────────
    agent, episode_rewards, episode_energies = train(cfg, args.episodes)

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\n  Running final evaluation …")
    rl_reward, rl_energy, rl_unnecessary         = run_rl_eval(agent, cfg)
    bl_reward, bl_energy, bl_unnecessary          = run_baseline(cfg)

    # ── Log to CSV ────────────────────────────────────────────────────────
    row = log_results(cfg, agent, episode_rewards, episode_energies,
                      rl_reward, rl_energy, bl_reward, bl_energy)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_dir = cfg["logging"]["plot_dir"]
    plot_reward_curve(episode_rewards, plot_dir)
    plot_energy_comparison(rl_energy, bl_energy,
                           rl_unnecessary, bl_unnecessary,
                           plot_dir, cfg["energy"]["cost_per_kwh"])
                           
    # Log plots to MLFlow if there is an active run (wait, start_run is handled in log_results, so we log artifacts there. Let's do it inside the MLFlow run context!)
    # Actually, to make it clean, let's just log the whole plot dir at the end.
    mlflow.set_experiment("Smart_Street_Lighting_RL")
    # Fetch the last active run ID to log plots
    last_run = mlflow.last_active_run()
    if last_run:
        with mlflow.start_run(run_id=last_run.info.run_id):
            mlflow.log_artifacts(plot_dir, "plots")

    # ── Summary table ─────────────────────────────────────────────────────
    print_comparison(row, rl_unnecessary, bl_unnecessary)

    print("  [DONE] All done! Check experiments/ for results and plots.\n")


if __name__ == "__main__":
    main()
