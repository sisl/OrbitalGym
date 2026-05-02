"""Tests for observations/reference.py — FullObservation."""

import jax
import jax.numpy as jnp

from orbital_game.env.types import Side
from orbital_game.observations.base import ObservationScope
from orbital_game.observations.reference import FullObservation


def test_full_observation_returns_flat_state_vector():
    class _Layout:
        def flatten(self, d, i):
            return jnp.concatenate([d, i])

    class _Env:
        guards = jnp.arange(6.0)
        bandits = jnp.arange(6.0) + 100

    obs_fn = FullObservation(layout=_Layout())
    out = obs_fn(
        env_state=_Env(),
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )
    assert jnp.allclose(out, jnp.concatenate([jnp.arange(6.0), jnp.arange(6.0) + 100]))


def test_full_observation_scope():
    class _Layout:
        def flatten(self, d, i):
            return jnp.concatenate([d, i])

    fn = FullObservation(layout=_Layout())
    assert fn.scope is ObservationScope.PER_SIDE


def test_full_observation_side_agnostic():
    """FullObservation returns the same result regardless of the side argument."""

    class _Layout:
        def flatten(self, d, i):
            return jnp.concatenate([d, i])

    class _Env:
        guards = jnp.arange(4.0)
        bandits = jnp.arange(4.0) + 50

    obs_fn = FullObservation(layout=_Layout())
    key = jax.random.PRNGKey(0)
    out_guard = obs_fn(env_state=_Env(), side=Side.GUARD, params=None, key=key, t=jnp.array(0.0))
    out_bandit = obs_fn(env_state=_Env(), side=Side.BANDIT, params=None, key=key, t=jnp.array(0.0))
    assert jnp.allclose(out_guard, out_bandit)
