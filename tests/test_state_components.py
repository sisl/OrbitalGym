"""Tests for state components: fields() shape contract and zeros(n) initialization."""
import jax.numpy as jnp
import pytest

from orbital_game.state.components import (
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
        (RTState,   {"rt":  (4,)}),
        (RTNState,  {"rtn": (6,)}),
        (Mass,      {"propellant_mass": ()}),
        (Power,     {"charge": ()}),
        (Attitude,  {"quat": (4,)}),
        (BodyRates, {"omega": (3,)}),
    ],
)
def test_component_fields_match_spec(component, expected_fields):
    assert dict(component.fields()) == expected_fields


@pytest.mark.parametrize(
    "component, field, expected_leaf_shape",
    [
        (RTState,   "rt",              (5, 4)),
        (RTNState,  "rtn",             (5, 6)),
        (Mass,      "propellant_mass", (5,)),
        (Power,     "charge",          (5,)),
        (Attitude,  "quat",            (5, 4)),
        (BodyRates, "omega",           (5, 3)),
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
    names = {c.name for c in (RTState, RTNState, Mass, Power, Attitude, BodyRates)}
    assert len(names) == 6
