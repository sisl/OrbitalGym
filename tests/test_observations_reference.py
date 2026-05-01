"""Tests for observations/reference.py — FullObservation."""

import jax
import jax.numpy as jnp

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
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )
    assert jnp.allclose(out, jnp.concatenate([jnp.arange(6.0), jnp.arange(6.0) + 100]))
