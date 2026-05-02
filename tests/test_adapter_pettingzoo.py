"""Tests for PettingZooAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from examples.reference_scenario import build_config
from orbital_game.env.core import OrbitalGameEnv

pettingzoo = pytest.importorskip("pettingzoo")


def _make_adapter():
    from orbital_game.adapters.pettingzoo import PettingZooAdapter
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    return PettingZooAdapter(env, seed=0)


def test_pz_adapter_constructs():
    adapter = _make_adapter()
    assert adapter.possible_agents == ["guard_0", "bandit_0"]


def test_pz_adapter_reset_returns_per_agent_dicts():
    adapter = _make_adapter()
    obs_dict, info_dict = adapter.reset(seed=0)
    assert set(obs_dict.keys()) == {"guard_0", "bandit_0"}
    assert set(info_dict.keys()) == {"guard_0", "bandit_0"}
    for v in obs_dict.values():
        assert isinstance(v, np.ndarray)
        assert v.dtype == np.float32


def test_pz_adapter_step_returns_5_dicts():
    adapter = _make_adapter()
    adapter.reset(seed=0)
    action_space = adapter.action_space("guard_0")
    actions = {
        "guard_0": np.zeros(action_space.shape, dtype=np.float32),
        "bandit_0": np.zeros(action_space.shape, dtype=np.float32),
    }
    obs, reward, terminated, truncated, info = adapter.step(actions)
    assert set(obs.keys()) == {"guard_0", "bandit_0"}
    assert set(reward.keys()) == {"guard_0", "bandit_0"}
    assert set(terminated.keys()) == {"guard_0", "bandit_0"}
    assert set(truncated.keys()) == {"guard_0", "bandit_0"}
    assert all(isinstance(r, float) for r in reward.values())
    assert all(isinstance(t, bool) for t in terminated.values())


def test_pz_adapter_passes_parallel_api_test():
    """Run pettingzoo's parallel API test against the adapter."""
    from pettingzoo.test import parallel_api_test
    adapter = _make_adapter()
    parallel_api_test(adapter, num_cycles=3)


def test_pz_adapter_seed_reproducibility():
    adapter1 = _make_adapter()
    obs1, _ = adapter1.reset(seed=42)
    adapter2 = _make_adapter()
    obs2, _ = adapter2.reset(seed=42)
    for aid in obs1:
        np.testing.assert_array_equal(obs1[aid], obs2[aid])
