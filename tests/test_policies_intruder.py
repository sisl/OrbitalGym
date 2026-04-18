"""Tests for policies/intruder.py — reference intruder policy."""

import jax
import jax.numpy as jnp

from orbital_game.policies.intruder import ZeroControlIntruder


def test_zero_control_intruder_returns_zeros():
    pol = ZeroControlIntruder(n_intruders=3, action_dim=3)
    out = pol(obs=None, key=jax.random.PRNGKey(0), t=jnp.array(0.0))
    assert out.shape == (3, 3)
    assert jnp.allclose(out, 0.0)
