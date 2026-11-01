"""
Integration tests for RL training, checkpoint serialization, and comparative benchmark pipeline.
"""

import os
import numpy as np
import pytest

from simulation.environment import FodderMindGymEnv
from algorithms.ppo_agent import (
    create_ppo_agent,
    train_ppo_agent,
    save_ppo_agent,
    load_ppo_agent,
    make_vectorized_env,
)
from algorithms.rule_based import GymRuleBasedPolicy
from algorithms.evaluation import run_full_benchmark


def test_fast_ppo_training_and_checkpoint(tmp_path):
    env_fn = lambda: FodderMindGymEnv()
    vec_env = make_vectorized_env(env_fn, num_envs=1, seed=10)

    model = create_ppo_agent(
        env=vec_env,
        seed=10,
        n_steps=128,
        batch_size=32,
        n_epochs=2,
    )

    # Train for 256 steps (rapid CPU verification)
    train_ppo_agent(model, total_timesteps=256)

    # Test inference
    single_env = FodderMindGymEnv()
    obs, _ = single_env.reset(seed=10)
    action, _ = model.predict(obs, deterministic=True)

    assert action.shape == (5,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    # Save and reload
    save_path = str(tmp_path / "test_ppo.zip")
    save_ppo_agent(model, save_path)
    assert os.path.exists(save_path)

    loaded_model = load_ppo_agent(save_path, env=vec_env)
    action_loaded, _ = loaded_model.predict(obs, deterministic=True)
    np.testing.assert_allclose(action, action_loaded, rtol=1e-4)


def test_matched_benchmark_evaluation(tmp_path):
    env_fn = lambda scenario_name="baseline": FodderMindGymEnv(scenario_name=scenario_name)

    rule_policy = GymRuleBasedPolicy()

    # Create and train a small PPO agent
    vec_env = make_vectorized_env(env_fn, num_envs=1, seed=42)
    ppo_model = create_ppo_agent(env=vec_env, seed=42, n_steps=128, batch_size=32)
    train_ppo_agent(ppo_model, total_timesteps=256)

    # Execute matched benchmark
    out_dir = str(tmp_path / "benchmark_out")
    raw_df, summary_df = run_full_benchmark(
        rule_policy=rule_policy,
        ppo_policy=ppo_model,
        env_factory=env_fn,
        train_seeds=[101, 102],
        held_out_seeds=[901, 902],
        output_dir=out_dir,
    )

    assert len(raw_df) > 0
    assert "policy" in raw_df.columns
    assert "cumulative_reward" in raw_df.columns
    assert "total_feed_cost" in raw_df.columns
    assert "water_use_liters" in raw_df.columns
    assert "shortage_duration_days" in raw_df.columns

    # Verify CSV files exist
    assert os.path.exists(os.path.join(out_dir, "tables", "benchmark_rl_vs_rule.csv"))
    assert os.path.exists(os.path.join(out_dir, "tables", "benchmark_summary.csv"))

    # Verify Plots generated
    assert os.path.exists(os.path.join(out_dir, "figures", "benchmark_metrics_barchart.png"))
    assert os.path.exists(os.path.join(out_dir, "figures", "benchmark_trajectory_comparison.png"))
