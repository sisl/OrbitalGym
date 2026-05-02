"""Tests for belief/gaussian.py — GaussianFromTruth + GaussianUniformDefault initializers."""

import jax
import jax.numpy as jnp

from orbital_game.belief.gaussian import (
    GaussianFromTruthInitializer,
    GaussianUniformDefaultInitializer,
)
from orbital_game.env.types import Side


class _Layout:
    flat_dim = 4

    def flatten(self, d, i):
        return jnp.concatenate([d, i])


class _Env:
    guards = jnp.array([1.0, 2.0])
    bandits = jnp.array([3.0, 4.0])


def test_gaussian_from_truth_uses_env_state_as_mean():
    init = GaussianFromTruthInitializer(layout=_Layout(), variance_diag=jnp.ones(4))
    belief = init(env_state=_Env(), side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert jnp.allclose(belief.mean, jnp.array([1.0, 2.0, 3.0, 4.0]))
    assert jnp.allclose(belief.cov, jnp.eye(4))


def test_gaussian_uniform_default_ignores_env_state():
    init = GaussianUniformDefaultInitializer(
        default_mean=jnp.array([10.0, 20.0, 30.0, 40.0]),
        variance_diag=jnp.array([1.0, 2.0, 3.0, 4.0]),
    )
    belief = init(env_state=_Env(), side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert jnp.allclose(belief.mean, jnp.array([10.0, 20.0, 30.0, 40.0]))
    assert jnp.allclose(belief.cov, jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0])))
