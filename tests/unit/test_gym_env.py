"""
Unit tests for FodderMindGymEnv (Gymnasium standard compliance).
"""

import numpy as np
import pytest
from simulation.environment import FodderMindGymEnv


def test_gym_env_spaces():
    env = FodderMindGymEnv()
    obs, info = env.reset(seed=123)

    assert env.observation_space.shape == (16,)
    assert env.action_space.shape == (5,)
    assert obs.shape == (16,)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert "day" in info
    assert info["day"] == 0


def test_gym_env_step_contract():
    env = FodderMindGymEnv()
    env.reset(seed=42)

    # Valid step with zero action
    action = np.zeros(5, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (16,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "daily_feed_cost" in info
    assert "pen_weight_kg" in info


def test_deterministic_seed_reproducibility():
    env1 = FodderMindGymEnv()
    env2 = FodderMindGymEnv()

    obs1, _ = env1.reset(seed=999)
    obs2, _ = env2.reset(seed=999)

    np.testing.assert_allclose(obs1, obs2, rtol=1e-5, atol=1e-5)

    action = np.array([0.2, -0.1, 0.5, -0.4, 0.8], dtype=np.float32)
    next_obs1, r1, _, _, _ = env1.step(action)
    next_obs2, r2, _, _, _ = env2.step(action)

    np.testing.assert_allclose(next_obs1, next_obs2, rtol=1e-5, atol=1e-5)
    assert r1 == pytest.approx(r2, rel=1e-5)


def test_safety_shield_acidosis_intervention():
    env = FodderMindGymEnv()
    env.reset(seed=42)

    # Action heavily favoring concentrate (index 3), which would cause low NDF acidosis
    dangerous_action = np.array([-5.0, -5.0, -5.0, 5.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(dangerous_action)

    # The safety shield must intervene and prevent acidosis
    assert bool(info["shield_intervened"]) is True
    # Dietary NDF should not trigger severe acidosis risk
    assert bool(info["acidosis_risk"]) is False
