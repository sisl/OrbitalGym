"""Tests for ZeroControl — role-agnostic policy library."""

import jax
import jax.numpy as jnp

from orbital_game.policies import ZeroControl
from tests.policies._helpers import make_impulsive_maneuver_command_cls


def test_zero_control_returns_zero_command():
    cmd_cls = make_impulsive_maneuver_command_cls(3)
    pol = ZeroControl(n_vehicles=3, command_cls=cmd_cls)
    cmd, next_ps = pol(None, obs=None, key=jax.random.PRNGKey(0), t=jnp.array(0.0))
    assert cmd.dv.shape == (3, 3)
    assert jnp.allclose(cmd.dv, 0.0)
    assert next_ps is None


def test_zero_control_replace_dims():
    """ZeroControl's n_vehicles/command_cls are populated via dataclasses.replace."""
    import dataclasses

    bare = ZeroControl()
    assert bare.n_vehicles == 0 and bare.command_cls is None

    cmd_cls = make_impulsive_maneuver_command_cls(3)
    bound = dataclasses.replace(bare, n_vehicles=3, command_cls=cmd_cls)
    assert bound.n_vehicles == 3 and bound.command_cls is cmd_cls

    cmd, ps = bound(None, None, None, None)
    assert cmd.dv.shape == (3, 3)
    assert ps is None
