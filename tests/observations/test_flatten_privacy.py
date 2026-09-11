"""Invisible sensor payloads must never enter flat policy observations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym.adapters.gymnasium import GymnasiumAdapter
from orbitalgym.adapters.pettingzoo import PettingZooAdapter
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Side
from orbitalgym.observations.composite import CompositeObservation
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.observations.onboard_gps import OnboardGPSObservation
from orbitalgym.observations.types import flatten_observations, flatten_observations_per_agent
from tests.eval.test_information_metrics import _cone_pointed_away_cfg, _pin_geometry
from tests.observations.test_observations_conical import _layout_rtn, _state_rtn


@pytest.mark.parametrize("per_agent", [False, True])
def test_conical_hidden_truth_is_masked_per_channel_under_jit_vmap(per_agent):
    layout = _layout_rtn(n_self=2, n_tgt=2)
    sensor = CompositeObservation(
        constituents=(
            OnboardGPSObservation(layout, sigma_gps=0.0),
            ConicalObservation(layout, jnp.array([[1.0, 0.0, 0.0]]), 0.2, 1.0, 0.01),
        )
    )
    flatten = flatten_observations_per_agent if per_agent else flatten_observations

    def observe(hidden_x):
        state = _state_rtn(
            jnp.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            jnp.array([[100.0, 0.0, 0.0], [hidden_x, 0.0, 0.0]]),
        )
        channels = sensor(state, None, Side.GUARD, None, jax.random.PRNGKey(0), 0.0)
        return flatten(channels)

    flat = jax.jit(jax.vmap(observe))(jnp.array([-100.0, -2000.0]))
    np.testing.assert_array_equal(flat[0], flat[1])
    expected_shape = (2, 2, 48) if per_agent else (2, 96)
    assert flat.shape == expected_shape
    # Inspect each channel separately: GPS visibility must not reveal the
    # cone's own-side payload, nor cone visibility reveal GPS noise.
    state = _state_rtn(
        jnp.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        jnp.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]),
    )
    channels = sensor(state, None, Side.GUARD, None, jax.random.PRNGKey(0), 0.0)
    for channel in channels:
        output = flatten((channel,)).reshape(channel.obs.shape)
        np.testing.assert_array_equal(output[~channel.visible], 0.0)
        np.testing.assert_array_equal(output[channel.visible], channel.obs[channel.visible])


@pytest.mark.parametrize("adapter_kind", ["gymnasium", "pettingzoo", "pomdp"])
def test_adapters_do_not_publish_an_unseen_conical_target(adapter_kind):
    cfg = _cone_pointed_away_cfg()
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    state = _pin_geometry(state)
    if adapter_kind == "gymnasium":
        adapter = GymnasiumAdapter(env)
        adapter.reset(seed=0)
        adapter._state = state
        obs, *_ = adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))
    elif adapter_kind == "pettingzoo":
        adapter = PettingZooAdapter(env)
        adapter.reset(seed=0)
        adapter._state = state
        observations, *_ = adapter.step(
            {
                agent: np.zeros(adapter.action_space(agent).shape, dtype=np.float32)
                for agent in adapter.agents
            }
        )
        obs = observations["guard_0"]
    else:
        adapter = POMDPAdapter(env)
        packed = adapter._pack(state)
        action = jnp.zeros(adapter.guard_action_flat_dim + adapter.bandit_action_flat_dim)
        obs = adapter.observation(packed, action, packed, Side.GUARD)
    assert obs.shape == (8,)
    np.testing.assert_array_equal(obs, 0.0)


def test_model_state_merge_retains_visible_gps_and_conical_measurements():
    from orbitalgym.observations import types
    from orbitalgym.observations.teammate_ephemeris import TeammateEphemerisObservation

    layout = _layout_rtn(n_self=2, n_tgt=2)
    state = _state_rtn(
        jnp.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        jnp.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]),
    )
    sensor = CompositeObservation(
        constituents=(
            OnboardGPSObservation(layout, sigma_gps=0.0),
            TeammateEphemerisObservation(layout, sigma=0.0),
            ConicalObservation(layout, jnp.array([[1.0, 0.0, 0.0]]), 0.2, 0.0, 0.0),
        )
    )
    channels = sensor(state, None, Side.GUARD, None, jax.random.PRNGKey(0), 0.0)
    result = jax.jit(lambda: types.merge_full_state_observations(channels, 6))()
    expected = np.zeros((2, 4, 6))
    expected[:, :, 0] = [[10.0, 20.0, 100.0, 0.0], [10.0, 20.0, 100.0, 0.0]]
    np.testing.assert_array_equal(result.reshape(2, 4, 6), expected)


def test_model_state_merge_uses_masks_even_when_a_visible_measurement_is_zero():
    from orbitalgym.observations import types

    channel = types.Observation(
        obs=jnp.ones((1, 2, 4)),
        visible=jnp.ones((1, 2), dtype=bool),
        obs_matrix=jnp.eye(4),
        obs_noise=jnp.eye(4),
    )
    last = channel.replace(obs=jnp.zeros((1, 2, 4)), visible=jnp.array([[True, False]]))
    result = types.merge_full_state_observations((channel, last), 4)
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])


def test_model_state_merge_rejects_position_only_channels():
    from orbitalgym.observations import types

    channel = types.Observation(
        obs=jnp.ones((1, 2, 3)),
        visible=jnp.ones((1, 2), dtype=bool),
        obs_matrix=jnp.eye(3, 6),
        obs_noise=jnp.eye(3),
    )
    with pytest.raises(ValueError, match="full-state"):
        types.merge_full_state_observations((channel,), 6)
