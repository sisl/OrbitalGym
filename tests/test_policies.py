"""Tests for policies/bandit.py — reference bandit policy."""

import jax
import jax.numpy as jnp

from orbital_game.policies.bandit import ZeroControlBandit


def test_zero_control_bandit_returns_zeros():
    pol = ZeroControlBandit(n_bandits=3, action_dim=3)
    out = pol(obs=None, key=jax.random.PRNGKey(0), t=jnp.array(0.0))
    assert out.shape == (3, 3)
    assert jnp.allclose(out, 0.0)
