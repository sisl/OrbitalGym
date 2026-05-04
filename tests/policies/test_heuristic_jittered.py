"""JitteredPolicy — wraps a base policy and adds PRNG-keyed Gaussian jitter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbital_game.policies import ZeroControl
from orbital_game.policies.heuristic import JitteredPolicy


def test_jittered_zero_control_emits_nonzero_action_with_finite_sigma():
    base = ZeroControl(n_vehicles=1, action_dim=3)
    p = JitteredPolicy(base=base, sigma=0.01, n_vehicles=1, action_dim=3)
    obs = jnp.zeros(12)
    action, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert action.shape == (1, 3)
    assert jnp.all(jnp.isfinite(action))
    # Non-zero with overwhelming probability for sigma=0.01.
    assert float(jnp.linalg.norm(action[0])) > 0.0


def test_jittered_zero_sigma_is_passthrough():
    base = ZeroControl(n_vehicles=1, action_dim=3)
    p = JitteredPolicy(base=base, sigma=0.0, n_vehicles=1, action_dim=3)
    obs = jnp.zeros(12)
    action, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert pytest.approx(float(jnp.linalg.norm(action[0])), abs=1e-12) == 0.0


def test_jittered_is_deterministic_for_same_key():
    base = ZeroControl(n_vehicles=1, action_dim=3)
    p = JitteredPolicy(base=base, sigma=0.01, n_vehicles=1, action_dim=3)
    obs = jnp.zeros(12)
    a1, _ = p(None, obs, jax.random.PRNGKey(42), jnp.asarray(0))
    a2, _ = p(None, obs, jax.random.PRNGKey(42), jnp.asarray(0))
    assert jnp.array_equal(a1, a2)
