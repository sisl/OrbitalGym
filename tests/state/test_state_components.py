"""Tests for state components: fields() shape contract and zeros(n) initialization."""

import jax.numpy as jnp
import pytest

from orbital_game.state.components import (
    AppliedDV,
    AppliedTorque,
    Attitude,
    BodyRates,
    Mass,
    Power,
    RTNState,
    RTState,
)


@pytest.mark.parametrize(
    "component, expected_fields",
    [
        (RTState, {"rt": (4,)}),
        (RTNState, {"rtn": (6,)}),
        (Mass, {"propellant_mass": ()}),
        (Power, {"charge": ()}),
        (Attitude, {"quat": (4,)}),
        (BodyRates, {"omega": (3,)}),
        (AppliedDV, {"applied_dv": (3,)}),
        (AppliedTorque, {"applied_torque": (3,)}),
    ],
)
def test_component_fields_match_spec(component, expected_fields):
    assert dict(component.fields()) == expected_fields


@pytest.mark.parametrize(
    "component, field, expected_leaf_shape",
    [
        (RTState, "rt", (5, 4)),
        (RTNState, "rtn", (5, 6)),
        (Mass, "propellant_mass", (5,)),
        (Power, "charge", (5,)),
        (Attitude, "quat", (5, 4)),
        (BodyRates, "omega", (5, 3)),
        (AppliedDV, "applied_dv", (5, 3)),
        (AppliedTorque, "applied_torque", (5, 3)),
    ],
)
def test_component_zeros_shapes(component, field, expected_leaf_shape):
    zeros = component.zeros(5)
    assert field in zeros
    assert zeros[field].shape == expected_leaf_shape
    # Attitude initializes to identity quaternion, not zeros
    if component is Attitude:
        assert jnp.allclose(zeros[field][:, 0], 1.0)  # w component = 1
        assert jnp.allclose(zeros[field][:, 1:], 0.0)  # x, y, z components = 0
    else:
        assert jnp.all(zeros[field] == 0.0)


def test_component_names_unique():
    names = {
        c.name
        for c in (RTState, RTNState, Mass, Power, Attitude, BodyRates, AppliedDV, AppliedTorque)
    }
    assert len(names) == 8


def test_eci_state_zeros():
    from orbital_game.state.components import ECIState

    z = ECIState.zeros(3)
    assert z["eci"].shape == (3, 6)
    assert (z["eci"] == 0).all()


def test_eci_state_component_key():
    from orbital_game.registry import StateComponentKey

    assert StateComponentKey.ECI.value == "eci"


def test_eci_state_assemble():
    from orbital_game.state.assemble import build_state_class
    from orbital_game.state.components import ECIState

    cls = build_state_class([ECIState], n_vehicles=2, class_name="EciOnly")
    inst = cls.zeros(2)
    assert inst.eci.shape == (2, 6)


def test_applied_dv_fields():
    assert AppliedDV.fields() == {"applied_dv": (3,)}


def test_applied_dv_zeros_shape():
    z = AppliedDV.zeros(5)
    assert z["applied_dv"].shape == (5, 3)
    assert jnp.all(z["applied_dv"] == 0.0)


def test_applied_torque_fields():
    assert AppliedTorque.fields() == {"applied_torque": (3,)}


def test_applied_torque_zeros_shape():
    z = AppliedTorque.zeros(4)
    assert z["applied_torque"].shape == (4, 3)
    assert jnp.all(z["applied_torque"] == 0.0)
