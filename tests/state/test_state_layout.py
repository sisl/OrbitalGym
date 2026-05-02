"""Tests for StateLayout: per-vehicle accessors + flat view round-trip."""

import jax.numpy as jnp
import pytest

from orbital_game.state.assemble import build_state_class
from orbital_game.state.components import Mass, RTNState
from orbital_game.state.layout import StateLayout


@pytest.fixture
def layout():
    GuardState_test = build_state_class([RTNState, Mass], n_vehicles=2, class_name="DS")  # noqa: N806
    BanditState_test = build_state_class([RTNState], n_vehicles=3, class_name="IS")  # noqa: N806
    return StateLayout.build(
        guard_state_cls=GuardState_test,
        bandit_state_cls=BanditState_test,
        n_guards=2,
        n_bandits=3,
    )


def test_individual_guard_accessor_returns_single_vehicle_pytree(layout):
    GuardState_test = layout.guard_state_cls  # noqa: N806
    guards = GuardState_test.zeros(2).replace(
        rtn=jnp.arange(12.0).reshape(2, 6),
        propellant_mass=jnp.array([10.0, 20.0]),
    )
    d0 = layout.guard(guards, 0)
    assert d0.rtn.shape == (6,)
    assert jnp.allclose(d0.rtn, jnp.arange(6.0))
    assert jnp.isclose(d0.propellant_mass, 10.0)


def test_individual_bandit_accessor(layout):
    BanditState_test = layout.bandit_state_cls  # noqa: N806
    bandits = BanditState_test.zeros(3).replace(rtn=jnp.arange(18.0).reshape(3, 6))
    i2 = layout.bandit(bandits, 2)
    assert i2.rtn.shape == (6,)
    assert jnp.allclose(i2.rtn, jnp.arange(12.0, 18.0))


def test_flatten_unflatten_roundtrip(layout):
    GuardState_test = layout.guard_state_cls  # noqa: N806
    BanditState_test = layout.bandit_state_cls  # noqa: N806
    guards = GuardState_test.zeros(2).replace(
        rtn=jnp.arange(12.0).reshape(2, 6),
        propellant_mass=jnp.array([10.0, 20.0]),
    )
    bandits = BanditState_test.zeros(3).replace(rtn=jnp.arange(18.0).reshape(3, 6) + 100)

    vec = layout.flatten(guards, bandits)
    restored_guards, restored_bandits = layout.unflatten(vec)

    assert jnp.allclose(restored_guards.rtn, guards.rtn)
    assert jnp.allclose(restored_guards.propellant_mass, guards.propellant_mass)
    assert jnp.allclose(restored_bandits.rtn, bandits.rtn)


def test_flat_dim_matches_total_leaf_count(layout):
    # 2 guards × (6 rtn + 1 mass) + 3 bandits × 6 rtn = 14 + 18 = 32
    assert layout.flat_dim == 32
