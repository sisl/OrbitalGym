"""Tests for GymnasiumAdapter."""

from __future__ import annotations

import numpy as np

from examples.reference_scenario import build_config
from orbitalgym.env.core import OrbitalGymEnv


def _make_adapter():
    from orbitalgym.adapters.gymnasium import GymnasiumAdapter

    cfg = build_config()
    env = OrbitalGymEnv(cfg)
    return GymnasiumAdapter(env, seed=0)


def test_gym_adapter_constructs():
    adapter = _make_adapter()
    assert adapter.action_space is not None
    assert adapter.observation_space is not None


def test_gym_adapter_reset_returns_numpy_obs_and_info():
    adapter = _make_adapter()
    obs, info = adapter.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    adapter2 = _make_adapter()
    obs2, _ = adapter2.reset(seed=0)
    np.testing.assert_array_equal(obs, obs2)


def test_gym_adapter_step_returns_5_tuple():
    adapter = _make_adapter()
    obs, _info = adapter.reset(seed=0)
    action = np.zeros(adapter.action_space.shape, dtype=np.float32)
    out = adapter.step(action)
    assert len(out) == 5
    next_obs, reward, terminated, truncated, info = out
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_gym_adapter_step_advances_obs():
    adapter = _make_adapter()
    obs, _ = adapter.reset(seed=0)
    action = np.zeros(adapter.action_space.shape, dtype=np.float32)
    next_obs, _r, _term, _trunc, _info = adapter.step(action)
    assert not np.array_equal(obs, next_obs)


def test_gym_adapter_step_before_reset_raises():
    adapter = _make_adapter()
    action = np.zeros(adapter.action_space.shape, dtype=np.float32)
    try:
        adapter.step(action)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_gym_adapter_passes_check_env():
    """Run gymnasium's environment-checker against the adapter."""
    from gymnasium.utils.env_checker import check_env

    adapter = _make_adapter()
    check_env(adapter, skip_render_check=True, skip_close_check=True)
