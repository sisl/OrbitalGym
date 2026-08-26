"""JitteredPolicy — wraps a base policy and adds PRNG-keyed Gaussian jitter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbitalgym.policies import ZeroControl
from orbitalgym.policies.heuristic import JitteredPolicy
from tests.policies._helpers import make_impulsive_maneuver_command_cls


def test_jittered_zero_control_emits_nonzero_action_with_finite_sigma():
    cmd_cls = make_impulsive_maneuver_command_cls(1)
    base = ZeroControl(n_vehicles=1, command_cls=cmd_cls)
    p = JitteredPolicy(base=base, sigma=0.01, n_vehicles=1, command_cls=cmd_cls)
    obs = jnp.zeros(12)
    cmd, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert cmd.dv.shape == (1, 3)
    assert jnp.all(jnp.isfinite(cmd.dv))
    # Non-zero with overwhelming probability for sigma=0.01.
    assert float(jnp.linalg.norm(cmd.dv[0])) > 0.0


def test_jittered_zero_sigma_is_passthrough():
    cmd_cls = make_impulsive_maneuver_command_cls(1)
    base = ZeroControl(n_vehicles=1, command_cls=cmd_cls)
    p = JitteredPolicy(base=base, sigma=0.0, n_vehicles=1, command_cls=cmd_cls)
    obs = jnp.zeros(12)
    cmd, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert pytest.approx(float(jnp.linalg.norm(cmd.dv[0])), abs=1e-12) == 0.0


def test_jittered_is_deterministic_for_same_key():
    cmd_cls = make_impulsive_maneuver_command_cls(1)
    base = ZeroControl(n_vehicles=1, command_cls=cmd_cls)
    p = JitteredPolicy(base=base, sigma=0.01, n_vehicles=1, command_cls=cmd_cls)
    obs = jnp.zeros(12)
    a1, _ = p(None, obs, jax.random.PRNGKey(42), jnp.asarray(0))
    a2, _ = p(None, obs, jax.random.PRNGKey(42), jnp.asarray(0))
    assert jnp.array_equal(a1.dv, a2.dv)
