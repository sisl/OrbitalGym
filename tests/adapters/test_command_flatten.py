"""Round-trip test for the shared Command flatten/unflatten helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from orbital_game.actions.assemble import build_command_class
from orbital_game.actions.components import Communicate, ImpulsiveManeuver
from orbital_game.adapters._command_flatten import (
    command_flat_dim,
    flatten_command,
    unflatten_command,
)
from orbital_game.registry import Frame


def _make_impulsive_maneuver_command_cls(n: int):
    maneuver = ImpulsiveManeuver(
        action_frame=Frame.RT,
        truth_frame=Frame.RT,
        track_mass=False,
    )
    return build_command_class((maneuver,), n, "TestCommand")


def test_command_flat_dim_matches_field_layout():
    cmd_cls = _make_impulsive_maneuver_command_cls(n=4)
    # RT action_frame → dv is 2-D per agent, 4 agents → 4*2 = 8
    assert command_flat_dim(cmd_cls) == 8


def test_unflatten_flatten_round_trip_is_identity():
    cmd_cls = _make_impulsive_maneuver_command_cls(n=3)
    flat = jnp.arange(command_flat_dim(cmd_cls), dtype=jnp.float32)
    cmd = unflatten_command(cmd_cls, flat)
    flat_round_trip = flatten_command(cmd)
    assert jnp.array_equal(flat, flat_round_trip)


def test_flatten_unflatten_preserves_pytree_structure():
    cmd_cls = _make_impulsive_maneuver_command_cls(n=2)
    zeros_cmd = cmd_cls.zeros(2)
    flat = flatten_command(zeros_cmd)
    rebuilt = unflatten_command(cmd_cls, flat)
    # Pytree structure must match (flax dataclass equality via tree leaves).
    leaves_a = jtu.tree_leaves(zeros_cmd)
    leaves_b = jtu.tree_leaves(rebuilt)
    assert len(leaves_a) == len(leaves_b)
    for a, b in zip(leaves_a, leaves_b, strict=True):
        assert jnp.array_equal(a, b)


def test_unflatten_yields_correct_per_agent_shape():
    cmd_cls = _make_impulsive_maneuver_command_cls(n=5)
    flat = jnp.zeros(command_flat_dim(cmd_cls))
    cmd = unflatten_command(cmd_cls, flat)
    # RT action_frame → 2-D dv per agent.
    assert cmd.dv.shape == (5, 2)


def test_round_trip_under_jit():
    cmd_cls = _make_impulsive_maneuver_command_cls(n=2)
    flat = jax.random.normal(jax.random.PRNGKey(0), (command_flat_dim(cmd_cls),))

    @jax.jit
    def round_trip(f):
        return flatten_command(unflatten_command(cmd_cls, f))

    out = round_trip(flat)
    assert jnp.allclose(out, flat)


def test_communicate_active_dtype_round_trips():
    """`Communicate.active` is bool. Concatenating with float `dv`/`payload`
    upcasts it to float; unflatten must restore the declared bool dtype so
    consumers (CommsLeak observation, LBG comms reward) read truthy values
    correctly through Gym/PettingZoo/POMDP adapters.
    """
    maneuver = ImpulsiveManeuver(
        action_frame=Frame.RT,
        truth_frame=Frame.RT,
        track_mass=False,
    )
    cmd_cls = build_command_class((maneuver, Communicate()), 2, "TestCommandWithComms")
    cmd = cmd_cls.zeros(2).replace(
        # RT action_frame → 2-D dv.
        dv=jnp.array([[0.1, 0.2], [0.4, 0.5]]),
        active=jnp.array([True, False]),
        payload=jnp.zeros((2, 6)),
    )
    flat = flatten_command(cmd)
    rebuilt = unflatten_command(cmd_cls, flat)
    assert rebuilt.active.dtype == jnp.bool_
    assert bool(rebuilt.active[0]) is True
    assert bool(rebuilt.active[1]) is False
    # Float fields should still round-trip values.
    assert jnp.allclose(rebuilt.dv, cmd.dv)
    assert jnp.allclose(rebuilt.payload, cmd.payload)
