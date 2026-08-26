"""Tests for observations/reference.py — FullObservation (new tuple-of-channels contract)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbitalgym.env.types import Side
from orbitalgym.observations.reference import FullObservation
from orbitalgym.observations.types import Observation


class _Layout:
    """Stub layout exposing per-side fleet sizes and dynamics state dim."""

    def __init__(self, n_guards, n_bandits, d):
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


class _Env:
    def __init__(self, guards_rtn, bandits_rtn):
        from types import SimpleNamespace

        self.guards = SimpleNamespace(rtn=guards_rtn)
        self.bandits = SimpleNamespace(rtn=bandits_rtn)


def test_full_observation_returns_single_channel_tuple():
    n_guards = 2
    n_bandits = 3
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    guards_rtn = jnp.arange(n_guards * d, dtype=jnp.float64).reshape(n_guards, d)
    bandits_rtn = jnp.arange(n_bandits * d, dtype=jnp.float64).reshape(n_bandits, d) + 100.0
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)

    fn = FullObservation(layout=layout)
    out = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )

    assert isinstance(out, tuple)
    assert len(out) == 1
    ch = out[0]
    assert isinstance(ch, Observation)
    # Shape: (N_obs=N_self_for_side, N_total=N_self+N_tgt, d)
    assert ch.obs.shape == (n_guards, n_guards + n_bandits, d)
    assert ch.visible.shape == (n_guards, n_guards + n_bandits)
    assert ch.obs_matrix.shape == (d, d)
    assert ch.obs_noise.shape == (d, d)
    # All visible.
    assert bool(jnp.all(ch.visible))
    # Per pair (i, k): obs[i, k] == truth of entity k.
    for i in range(n_guards):
        for k in range(n_guards):
            assert jnp.allclose(ch.obs[i, k], guards_rtn[k])
        for j in range(n_bandits):
            assert jnp.allclose(ch.obs[i, n_guards + j], bandits_rtn[j])


def test_full_observation_side_swaps_own_and_opposing():
    n_guards = 1
    n_bandits = 2
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    guards_rtn = jnp.ones((n_guards, d))
    bandits_rtn = jnp.ones((n_bandits, d)) * 2.0
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)

    fn = FullObservation(layout=layout)
    out_g = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )
    out_b = fn(
        env_state=env,
        actions=None,
        side=Side.BANDIT,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )

    # Guard side: own (1) first, then bandits (2). N_obs=1, N_total=3.
    assert out_g[0].obs.shape == (1, 3, d)
    # Bandit side: own (2) first, then guards (1). N_obs=2, N_total=3.
    assert out_b[0].obs.shape == (2, 3, d)
