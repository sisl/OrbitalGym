"""Tests for StateLayout: per-vehicle accessors + flat view round-trip."""

import jax.numpy as jnp
import pytest

from orbital_game.state.assemble import build_state_class
from orbital_game.state.components import Mass, RTNState
from orbital_game.state.layout import StateLayout


@pytest.fixture
def layout():
    DefState = build_state_class([RTNState, Mass], n_vehicles=2, class_name="DS")  # noqa: N806
    IntState = build_state_class([RTNState], n_vehicles=3, class_name="IS")  # noqa: N806
    return StateLayout.build(
        defender_state_cls=DefState,
        intruder_state_cls=IntState,
        n_defenders=2,
        n_intruders=3,
    )


def test_individual_defender_accessor_returns_single_vehicle_pytree(layout):
    DefState = layout.defender_state_cls  # noqa: N806
    defs = DefState.zeros(2).replace(
        rtn=jnp.arange(12.0).reshape(2, 6),
        propellant_mass=jnp.array([10.0, 20.0]),
    )
    d0 = layout.defender(defs, 0)
    assert d0.rtn.shape == (6,)
    assert jnp.allclose(d0.rtn, jnp.arange(6.0))
    assert jnp.isclose(d0.propellant_mass, 10.0)


def test_individual_intruder_accessor(layout):
    IntState = layout.intruder_state_cls  # noqa: N806
    ints = IntState.zeros(3).replace(rtn=jnp.arange(18.0).reshape(3, 6))
    i2 = layout.intruder(ints, 2)
    assert i2.rtn.shape == (6,)
    assert jnp.allclose(i2.rtn, jnp.arange(12.0, 18.0))


def test_flatten_unflatten_roundtrip(layout):
    DefState = layout.defender_state_cls  # noqa: N806
    IntState = layout.intruder_state_cls  # noqa: N806
    defs = DefState.zeros(2).replace(
        rtn=jnp.arange(12.0).reshape(2, 6),
        propellant_mass=jnp.array([10.0, 20.0]),
    )
    ints = IntState.zeros(3).replace(rtn=jnp.arange(18.0).reshape(3, 6) + 100)

    vec = layout.flatten(defs, ints)
    restored_defs, restored_ints = layout.unflatten(vec)

    assert jnp.allclose(restored_defs.rtn, defs.rtn)
    assert jnp.allclose(restored_defs.propellant_mass, defs.propellant_mass)
    assert jnp.allclose(restored_ints.rtn, ints.rtn)


def test_flat_dim_matches_total_leaf_count(layout):
    # 2 defenders × (6 rtn + 1 mass) + 3 intruders × 6 rtn = 14 + 18 = 32
    assert layout.flat_dim == 32
