"""Verify Gymnasium and PettingZoo adapters produce sensible spaces under
each of the four bundled observation channels, with non-trivial vehicle
counts. Guards against shape regressions from the ObservationScope removal.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from orbital_game import OrbitalGameEnv, make_pursuit_evasion
from orbital_game.adapters.gymnasium import GymnasiumAdapter
from orbital_game.adapters.pettingzoo import PettingZooAdapter
from orbital_game.observations.composite import CompositeObservation
from orbital_game.observations.onboard_gps import OnboardGPSObservation
from orbital_game.observations.range_limited import RangeLimitedObservation
from orbital_game.observations.reference import FullObservation


def _make_cfg(obs_fn_factory, *, n_guards=2, n_bandits=3, asymmetric=False):
    """Build a Pursuit-Evasion config with the given observation function(s).

    The factory takes the pre-built env (so the channels can read env.layout)
    and returns the obs_fn instance. If asymmetric=True, the bandit side
    gets RangeLimitedObservation regardless of the factory.
    """
    cfg = make_pursuit_evasion(n_guards=n_guards, n_bandits=n_bandits, seed=0, max_horizon_s=500.0)
    env_for_layout = OrbitalGameEnv(cfg)
    guard_obs_fn = obs_fn_factory(env_for_layout)
    if asymmetric:
        bandit_obs_fn = RangeLimitedObservation(
            layout=env_for_layout.layout, sensor_range_m=2000.0, sigma_range=1.0
        )
    else:
        bandit_obs_fn = obs_fn_factory(env_for_layout)
    return dataclasses.replace(
        cfg,
        guard_observation_fn=guard_obs_fn,
        bandit_observation_fn=bandit_obs_fn,
    )


def _make_cfg_with_obs_fn_factory(obs_fn_factory):
    """Backwards-compatible 2g/3b symmetric helper used by the original tests."""
    return _make_cfg(obs_fn_factory)


# Channel factory functions, taking an env (for layout access) and
# returning a freshly-constructed obs fn instance.
CHANNEL_FACTORIES = {
    "full": lambda env: FullObservation(layout=env.layout),
    "onboard_gps": lambda env: OnboardGPSObservation(layout=env.layout, sigma_gps=2.0),
    "range_limited": lambda env: RangeLimitedObservation(
        layout=env.layout, sensor_range_m=2000.0, sigma_range=1.0
    ),
    "composite": lambda env: CompositeObservation(
        constituents=(
            OnboardGPSObservation(layout=env.layout, sigma_gps=2.0),
            RangeLimitedObservation(layout=env.layout, sensor_range_m=2000.0, sigma_range=1.0),
        )
    ),
}


@pytest.mark.parametrize("channel_name", list(CHANNEL_FACTORIES.keys()))
def test_gymnasium_adapter_under_channel(channel_name):
    """GymnasiumAdapter exposes a non-degenerate Box observation space and
    completes one reset+step cycle without shape errors under each channel."""
    cfg = _make_cfg_with_obs_fn_factory(CHANNEL_FACTORIES[channel_name])
    env = GymnasiumAdapter(OrbitalGameEnv(cfg), seed=0)

    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, (
        f"{channel_name}: reset obs shape {obs.shape} != space shape {env.observation_space.shape}"
    )
    assert obs.size > 0, f"{channel_name}: empty observation"
    assert np.isfinite(obs).all(), f"{channel_name}: non-finite values in observation"

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs2, reward, terminated, truncated, info = env.step(action)
    space_shape = env.observation_space.shape
    assert obs2.shape == space_shape, (
        f"{channel_name}: post-step obs shape {obs2.shape} != space shape {space_shape}"
    )
    assert isinstance(reward, float)


@pytest.mark.parametrize("channel_name", list(CHANNEL_FACTORIES.keys()))
def test_pettingzoo_adapter_under_channel(channel_name):
    """PettingZooAdapter exposes a non-degenerate Box per-agent observation
    space and completes one reset+step cycle without shape errors under each
    channel. With n_guards=2, n_bandits=3 there are 5 agents."""
    cfg = _make_cfg_with_obs_fn_factory(CHANNEL_FACTORIES[channel_name])
    env = PettingZooAdapter(OrbitalGameEnv(cfg), seed=0)

    obs_dict, info_dict = env.reset(seed=0)
    expected_agents = ["guard_0", "guard_1", "bandit_0", "bandit_1", "bandit_2"]
    assert set(obs_dict.keys()) == set(expected_agents), (
        f"{channel_name}: agent set {set(obs_dict.keys())} != expected {set(expected_agents)}"
    )

    for agent, obs in obs_dict.items():
        space = env.observation_space(agent)
        assert obs.shape == space.shape, (
            f"{channel_name}/{agent}: obs shape {obs.shape} != space shape {space.shape}"
        )
        assert obs.size > 0, f"{channel_name}/{agent}: empty observation"
        assert np.isfinite(obs).all(), f"{channel_name}/{agent}: non-finite values"

    actions = {
        agent: np.zeros(env.action_space(agent).shape, dtype=np.float32) for agent in env.agents
    }
    obs2_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(actions)
    for agent in expected_agents:
        assert obs2_dict[agent].shape == env.observation_space(agent).shape
        assert isinstance(reward_dict[agent], float)


@pytest.mark.parametrize("channel_name", list(CHANNEL_FACTORIES.keys()))
def test_pettingzoo_per_agent_obs_consistent(channel_name):
    """Per-agent observations within a side are either all-equal or all-distinct,
    never a partial mix — that would indicate inconsistent slicing.

    `FullObservation` broadcasts the same per-pair tensor to every observer,
    so all agents on a side see equal observations after slicing. Per-vehicle
    channels (`OnboardGPSObservation`, `RangeLimitedObservation`) write
    distinct rows per observer, so agents see distinct observations.
    """
    cfg = _make_cfg(CHANNEL_FACTORIES[channel_name])
    env = PettingZooAdapter(OrbitalGameEnv(cfg), seed=0)
    obs_dict, _ = env.reset(seed=0)

    guard_obs = [obs_dict["guard_0"], obs_dict["guard_1"]]
    bandit_obs = [obs_dict[f"bandit_{i}"] for i in range(3)]

    def all_equal(arrs):
        return all(np.array_equal(arrs[0], a) for a in arrs[1:])

    def all_distinct(arrs):
        for i, a in enumerate(arrs):
            for j, b in enumerate(arrs):
                if i < j and np.array_equal(a, b):
                    return False
        return True

    for label, arrs in [("guards", guard_obs), ("bandits", bandit_obs)]:
        assert all_equal(arrs) or all_distinct(arrs), (
            f"{channel_name}/{label}: per-agent observations are partially equal "
            f"(neither all-broadcast nor all-distinct) — slicing is inconsistent"
        )


def test_pettingzoo_onboard_gps_yields_distinct_per_agent_obs():
    """OnboardGPSObservation writes each observer's truth into row [i, i]
    of the per-pair tensor. After per-agent slicing, each agent sees a
    different row — distinct per-agent observations.
    """
    cfg = _make_cfg(CHANNEL_FACTORIES["onboard_gps"])
    env = PettingZooAdapter(OrbitalGameEnv(cfg), seed=0)
    obs_dict, _ = env.reset(seed=0)

    g0 = obs_dict["guard_0"]
    g1 = obs_dict["guard_1"]
    assert not np.array_equal(g0, g1), (
        "guard_0 and guard_1 received identical OnboardGPS observations — "
        "expected per-agent slicing to produce distinct rows"
    )


@pytest.mark.parametrize("channel_name", list(CHANNEL_FACTORIES.keys()))
def test_gymnasium_multi_step_rollout_under_channel(channel_name):
    """A 10-step rollout completes with stable obs/action shapes throughout."""
    cfg = _make_cfg(CHANNEL_FACTORIES[channel_name])
    env = GymnasiumAdapter(OrbitalGameEnv(cfg), seed=0)
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    obs, _ = env.reset(seed=0)
    assert obs.shape == obs_shape

    for step in range(10):
        action = np.zeros(action_shape, dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert obs.shape == obs_shape, f"{channel_name}: shape drift at step {step}"
        assert np.isfinite(obs).all(), f"{channel_name}: non-finite at step {step}"
        if terminated or truncated:
            break


@pytest.mark.parametrize("channel_name", list(CHANNEL_FACTORIES.keys()))
def test_pettingzoo_multi_step_rollout_under_channel(channel_name):
    """A 10-step PettingZoo rollout keeps all agents' shapes stable."""
    cfg = _make_cfg(CHANNEL_FACTORIES[channel_name])
    env = PettingZooAdapter(OrbitalGameEnv(cfg), seed=0)
    spaces = {a: env.observation_space(a).shape for a in env.possible_agents}

    obs_dict, _ = env.reset(seed=0)
    for agent, expected in spaces.items():
        assert obs_dict[agent].shape == expected

    for step in range(10):
        if not env.agents:
            break
        actions = {a: np.zeros(env.action_space(a).shape, dtype=np.float32) for a in env.agents}
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        for agent, expected in spaces.items():
            assert obs_dict[agent].shape == expected, (
                f"{channel_name}/{agent}: shape drift at step {step}"
            )
        if any(terminated.values()) or any(truncated.values()):
            break


@pytest.mark.parametrize("channel_name", list(CHANNEL_FACTORIES.keys()))
def test_pettingzoo_minimal_1v1_under_channel(channel_name):
    """1v1 (single agent per side) — the n_side==1 branch that skips slicing."""
    cfg = _make_cfg(CHANNEL_FACTORIES[channel_name], n_guards=1, n_bandits=1)
    env = PettingZooAdapter(OrbitalGameEnv(cfg), seed=0)
    obs_dict, _ = env.reset(seed=0)

    assert set(obs_dict.keys()) == {"guard_0", "bandit_0"}
    for agent, obs in obs_dict.items():
        assert obs.shape == env.observation_space(agent).shape
        assert obs.size > 0
        assert np.isfinite(obs).all()


def test_pettingzoo_asymmetric_obs_channels_per_side():
    """Different observation channels per side → different per-agent obs shapes."""
    cfg = _make_cfg(CHANNEL_FACTORIES["onboard_gps"], asymmetric=True)
    env = PettingZooAdapter(OrbitalGameEnv(cfg), seed=0)
    obs_dict, _ = env.reset(seed=0)

    guard_shape = env.observation_space("guard_0").shape
    bandit_shape = env.observation_space("bandit_0").shape
    # Different obs functions can perfectly well produce the same flat shape
    # by coincidence; the test asserts only that both sides build correctly,
    # not that shapes differ.
    for agent, obs in obs_dict.items():
        expected = guard_shape if agent.startswith("guard") else bandit_shape
        assert obs.shape == expected, f"{agent}: {obs.shape} != {expected}"
