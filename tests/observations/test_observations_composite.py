"""Tests for CompositeObservation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.env.types import Side
from orbital_game.observations.composite import CompositeObservation
from orbital_game.observations.onboard_gps import OnboardGPSObservation
from orbital_game.observations.range_limited import RangeLimitedObservation


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


def test_composite_concatenates_constituent_channels():
    n_guards = 1
    n_bandits = 2
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    env = _Env(guards_rtn=jnp.zeros((n_guards, d)), bandits_rtn=jnp.zeros((n_bandits, d)))

    gps = OnboardGPSObservation(layout=layout, sigma_gps=0.1)
    rng = RangeLimitedObservation(layout=layout, sensor_range_m=10000.0, sigma_range=1.0)
    comp = CompositeObservation(constituents=(gps, rng))

    out = comp(
        env_state=env, side=Side.GUARD, params=None, key=jax.random.PRNGKey(0), t=jnp.array(0.0)
    )

    assert len(out) == 2
    assert out[0].obs.shape == (1, 3, 6)
    assert out[1].obs.shape == (1, 3, 3)


def test_composite_with_one_constituent_passes_through():
    n_guards = 1
    n_bandits = 1
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    env = _Env(guards_rtn=jnp.zeros((n_guards, d)), bandits_rtn=jnp.zeros((n_bandits, d)))

    gps = OnboardGPSObservation(layout=layout, sigma_gps=0.0)
    comp = CompositeObservation(constituents=(gps,))

    out_comp = comp(
        env_state=env, side=Side.GUARD, params=None, key=jax.random.PRNGKey(0), t=jnp.array(0.0)
    )
    out_direct = gps(
        env_state=env, side=Side.GUARD, params=None, key=jax.random.PRNGKey(0), t=jnp.array(0.0)
    )
    assert len(out_comp) == len(out_direct)
    for a, b in zip(out_comp, out_direct, strict=True):
        assert jnp.allclose(a.obs, b.obs)
        assert jnp.array_equal(a.visible, b.visible)
