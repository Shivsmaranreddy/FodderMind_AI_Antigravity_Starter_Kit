"""
evaluation.py - Rigorous matched-seed evaluation and benchmark suite.

Compares Rule-Based Extension Heuristic vs PPO on identical:
- Training scenarios (baseline weather & market distributions)
- Held-out test scenarios (unseen random seeds, drought shock, price inflation)

Evaluation Metrics:
- Total Feed Cost (Currency)
- Shortage Duration (Days in deficit/shortage)
- Water Use (Total Liters)
- Nutrition Violations (Acidosis risk days, energy deficit days)
- Cumulative Reward
- Performance Variability (Mean, Std, Min, Max across seeds)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation.environment import FodderMindGymEnv


def evaluate_policy_on_scenario(
    policy: Any,
    env_factory,
    seeds: List[int],
    scenario_name: str = "baseline",
    split_name: str = "test",
    policy_name: str = "Policy",
) -> List[Dict[str, Any]]:
    """
    Evaluates a policy across a deterministic list of seeds.
    """
    records = []

    for seed in seeds:
        env: FodderMindGymEnv = env_factory(scenario_name=scenario_name)
        obs, info = env.reset(seed=seed)

        done = False
        truncated = False
        ep_reward = 0.0
        shortage_days = 0
        acidosis_days = 0
        deficit_days = 0
        daily_weights = []
        daily_costs = []

        while not (done or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            if info.get("feed_shortage_dm", 0.0) > 0.05 or info.get("energy_deficit", False):
                shortage_days += 1
            if info.get("acidosis_risk", False):
                acidosis_days += 1
            if info.get("energy_deficit", False):
                deficit_days += 1

            daily_weights.append(info.get("pen_weight_kg", 0.0))
            daily_costs.append(info.get("daily_feed_cost", 0.0))

        initial_wt = env.initial_pen_weight
        final_wt = env.pen.total_pen_weight_kg
        weight_gain = final_wt - initial_wt

        records.append({
            "policy": policy_name,
            "scenario": scenario_name,
            "split": split_name,
            "seed": seed,
            "cumulative_reward": round(ep_reward, 2),
            "total_feed_cost": round(env.total_feed_expenditure, 2),
            "water_use_liters": round(env.total_water_consumed, 1),
            "shortage_duration_days": shortage_days,
            "acidosis_violations": acidosis_days,
            "energy_deficit_violations": deficit_days,
            "total_spoilage_dm": round(env.total_spoilage_lost_dm, 2),
            "final_pen_weight_kg": round(final_wt, 2),
            "total_gain_kg": round(weight_gain, 2),
            "trajectory_weights": daily_weights,
            "trajectory_costs": daily_costs,
        })
        env.close()

    return records


def run_full_benchmark(
    rule_policy: Any,
    ppo_policy: Any,
    env_factory,
    train_seeds: Optional[List[int]] = None,
    held_out_seeds: Optional[List[int]] = None,
    output_dir: str = "outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes full matched comparative protocol across train and held-out scenarios.
    Exports CSV tables and publication plots.
    """
    train_seeds = train_seeds or [101, 102, 103, 104, 105]
    held_out_seeds = held_out_seeds or [901, 902, 903, 904, 905]

    all_records: List[Dict[str, Any]] = []

    # 1. Training scenarios (Baseline)
    for pol, name in [(rule_policy, "Rule_Based"), (ppo_policy, "PPO_Agent")]:
        recs = evaluate_policy_on_scenario(
            policy=pol,
            env_factory=env_factory,
            seeds=train_seeds,
            scenario_name="baseline",
            split_name="train",
            policy_name=name,
        )
        all_records.extend(recs)

    # 2. Held-out scenarios (Unseen Seeds)
    for pol, name in [(rule_policy, "Rule_Based"), (ppo_policy, "PPO_Agent")]:
        recs = evaluate_policy_on_scenario(
            policy=pol,
            env_factory=env_factory,
            seeds=held_out_seeds,
            scenario_name="baseline",
            split_name="held_out_baseline",
            policy_name=name,
        )
        all_records.extend(recs)

    # 3. Held-out stress scenarios (Drought & Price Spike)
    for pol, name in [(rule_policy, "Rule_Based"), (ppo_policy, "PPO_Agent")]:
        recs_drought = evaluate_policy_on_scenario(
            policy=pol,
            env_factory=env_factory,
            seeds=held_out_seeds[:3],
            scenario_name="drought",
            split_name="held_out_drought",
            policy_name=name,
        )
        recs_price = evaluate_policy_on_scenario(
            policy=pol,
            env_factory=env_factory,
            seeds=held_out_seeds[:3],
            scenario_name="price_spike",
            split_name="held_out_price_spike",
            policy_name=name,
        )
        all_records.extend(recs_drought)
        all_records.extend(recs_price)

    # Convert to DataFrames
    raw_df = pd.DataFrame(all_records)
    # Drop complex trajectory lists for tabular export
    export_df = raw_df.drop(columns=["trajectory_weights", "trajectory_costs"])

    # Aggregate summary by (policy, split)
    summary_df = export_df.groupby(["policy", "split"]).agg({
        "cumulative_reward": ["mean", "std"],
        "total_feed_cost": ["mean", "std"],
        "shortage_duration_days": ["mean", "std"],
        "water_use_liters": ["mean", "std"],
        "acidosis_violations": ["mean", "std"],
        "total_gain_kg": ["mean", "std"],
    }).round(2)

    # Save CSV files
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    raw_csv_path = os.path.join(tables_dir, "benchmark_rl_vs_rule.csv")
    export_df.to_csv(raw_csv_path, index=False)

    summary_csv_path = os.path.join(tables_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_csv_path)

    # Generate Plots
    plot_benchmark_metrics(export_df, os.path.join(figures_dir, "benchmark_metrics_barchart.png"))
    plot_trajectory_comparison(raw_df, os.path.join(figures_dir, "benchmark_trajectory_comparison.png"))

    return export_df, summary_df


def plot_benchmark_metrics(df: pd.DataFrame, output_path: str):
    """Generates comparison bar charts with variability bars across splits."""
    splits = df["split"].unique()
    policies = df["policy"].unique()
    metrics = ["total_feed_cost", "shortage_duration_days", "water_use_liters", "cumulative_reward"]
    titles = ["Total Feed Cost (Currency)", "Shortage Duration (Days)", "Water Use (Liters)", "Cumulative Reward"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        x = np.arange(len(splits))
        width = 0.35

        for p_idx, pol in enumerate(policies):
            pol_data = df[df["policy"] == pol]
            means = [pol_data[pol_data["split"] == s][metric].mean() for s in splits]
            stds = [pol_data[pol_data["split"] == s][metric].std() for s in splits]
            # Replace NaNs with 0
            means = [0.0 if np.isnan(m) else m for m in means]
            stds = [0.0 if np.isnan(s) else s for s in stds]

            offset = (p_idx - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=pol, capsize=4, alpha=0.85)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("held_out_", "H_") for s in splits], rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trajectory_comparison(df: pd.DataFrame, output_path: str):
    """Plots daily weight gain trajectories across matched seeds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for pol in df["policy"].unique():
        # Baseline train split
        sub = df[(df["policy"] == pol) & (df["split"] == "train")]
        weights = [r for r in sub["trajectory_weights"] if len(r) > 0]
        if weights:
            mean_traj = np.mean(weights, axis=0)
            days = np.arange(len(mean_traj))
            ax1.plot(days, mean_traj, label=f"{pol} (Train)", linewidth=2)

        # Held-out baseline split
        sub_h = df[(df["policy"] == pol) & (df["split"] == "held_out_baseline")]
        weights_h = [r for r in sub_h["trajectory_weights"] if len(r) > 0]
        if weights_h:
            mean_traj_h = np.mean(weights_h, axis=0)
            days_h = np.arange(len(mean_traj_h))
            ax2.plot(days_h, mean_traj_h, label=f"{pol} (Held-out)", linestyle="--", linewidth=2)

    ax1.set_title("Training Growth Trajectories (Mean over Seeds)", fontweight="bold")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Pen Weight (kg)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.set_title("Held-Out Growth Trajectories (Mean over Seeds)", fontweight="bold")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Pen Weight (kg)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
