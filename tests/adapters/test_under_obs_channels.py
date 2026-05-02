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


def _make_cfg_with_obs_fn_factory(obs_fn_factory):
    """Build a 2-guard / 3-bandit Pursuit-Evasion config and swap both
    sides' observation functions to whatever the factory produces.

    The factory takes the pre-built env (so the channels can read
    env.layout) and returns the obs_fn instance.
    """
    cfg = make_pursuit_evasion(n_guards=2, n_bandits=3, seed=0, max_horizon_s=500.0)
    env_for_layout = OrbitalGameEnv(cfg)
    obs_fn = obs_fn_factory(env_for_layout)
    return dataclasses.replace(
        cfg,
        guard_observation_fn=obs_fn,
        bandit_observation_fn=obs_fn,
    )


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
