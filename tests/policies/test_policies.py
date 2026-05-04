"""Tests for ZeroControl — role-agnostic policy library."""

import jax
import jax.numpy as jnp

from orbital_game.policies import ZeroControl


def test_zero_control_returns_zeros():
    pol = ZeroControl(n_vehicles=3, action_dim=3)
    out, next_ps = pol(None, obs=None, key=jax.random.PRNGKey(0), t=jnp.array(0.0))
    assert out.shape == (3, 3)
    assert jnp.allclose(out, 0.0)
    assert next_ps is None


def test_zero_control_replace_dims():
    """ZeroControl's n_vehicles/action_dim are populated via dataclasses.replace."""
    import dataclasses

    from orbital_game.policies import ZeroControl

    bare = ZeroControl()
    assert bare.n_vehicles == 0 and bare.action_dim == 0

    bound = dataclasses.replace(bare, n_vehicles=3, action_dim=2)
    assert bound.n_vehicles == 3 and bound.action_dim == 2

    out, ps = bound(None, None, None, None)
    assert out.shape == (3, 2)
    assert ps is None
