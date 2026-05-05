"""Tests for OnboardGPSObservation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.env.types import Side
from orbital_game.observations.onboard_gps import OnboardGPSObservation
from orbital_game.observations.types import Observation


class _Layout:
    def __init__(self, n_guards, n_bandits, d):
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


class _Env:
    def __init__(self, guards_rtn, bandits_rtn):
        from types import SimpleNamespace

        self.guards = SimpleNamespace(rtn=guards_rtn)
        self.bandits = SimpleNamespace(rtn=bandits_rtn)


def test_onboard_gps_visible_only_for_self_pair():
    n_guards = 2
    n_bandits = 3
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    guards_rtn = jnp.arange(n_guards * d, dtype=jnp.float64).reshape(n_guards, d)
    bandits_rtn = jnp.arange(n_bandits * d, dtype=jnp.float64).reshape(n_bandits, d) + 100.0
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)

    fn = OnboardGPSObservation(layout=layout, sigma_gps=0.1)
    out = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )

    assert len(out) == 1
    ch = out[0]
    assert isinstance(ch, Observation)
    assert ch.visible.shape == (n_guards, n_guards + n_bandits)
    expected = jnp.eye(n_guards, n_guards + n_bandits, dtype=bool)
    assert jnp.array_equal(ch.visible, expected)


def test_onboard_gps_obs_at_self_pair_equals_own_truth_with_noise_variance_consistent_with_r():
    n_guards = 2
    n_bandits = 1
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    guards_rtn = jnp.zeros((n_guards, d))
    bandits_rtn = jnp.zeros((n_bandits, d))
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)
    sigma = 0.5

    fn = OnboardGPSObservation(layout=layout, sigma_gps=sigma)
    key = jax.random.PRNGKey(42)
    out = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=key,
        t=jnp.array(0.0),
    )
    ch = out[0]

    assert jnp.allclose(ch.obs_noise, jnp.eye(d) * sigma**2)
    assert jnp.allclose(ch.obs_matrix, jnp.eye(d))
    assert bool(jnp.all(jnp.isfinite(ch.obs[0, 0])))
    assert bool(jnp.all(jnp.isfinite(ch.obs[1, 1])))


def test_onboard_gps_zero_noise_returns_truth_at_self_pair():
    n_guards = 2
    n_bandits = 1
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    guards_rtn = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    bandits_rtn = jnp.zeros((n_bandits, d))
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)

    fn = OnboardGPSObservation(layout=layout, sigma_gps=0.0)
    out = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )
    ch = out[0]

    assert jnp.allclose(ch.obs[0, 0], guards_rtn[0])
    assert jnp.allclose(ch.obs[1, 1], guards_rtn[1])
