"""
ppo_agent.py - Lightweight Stable-Baselines3 PPO agent for FodderMind AI.

Configured specifically for low-overhead CPU execution:
- Lightweight MLP policy head: net_arch = dict(pi=[64, 64], vf=[64, 64]).
- Bounded batch sizes and deterministic seeding.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def make_vectorized_env(env_factory, num_envs: int = 1, seed: int = 42):
    """Wraps environment factory into DummyVecEnv for CPU training."""
    def _init(idx: int):
        def _thunk():
            e = env_factory()
            e.reset(seed=seed + idx)
            return e
        return _thunk

    return DummyVecEnv([_init(i) for i in range(num_envs)])


def create_ppo_agent(
    env: gym.Env | DummyVecEnv,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 4,
    gamma: float = 0.99,
    verbose: int = 0,
) -> PPO:
    """
    Initializes a CPU-friendly PPO agent.
    """
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=verbose,
        device="cpu",
    )
    return model


def train_ppo_agent(
    model: PPO,
    total_timesteps: int = 10000,
    progress_bar: bool = False,
) -> PPO:
    """Trains PPO agent over the specified number of environment interactions."""
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
    return model


def save_ppo_agent(model: PPO, filepath: str) -> str:
    """Saves trained PPO weights to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    model.save(filepath)
    return filepath


def load_ppo_agent(filepath: str, env: Optional[gym.Env] = None) -> PPO:
    """Loads a serialized PPO agent checkpoint."""
    return PPO.load(filepath, env=env, device="cpu")
