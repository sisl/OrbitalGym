"""Tests for generic frozen-dataclass-with-enum-key serialization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import jax.numpy as jnp
import pytest

from orbital_game.registry import _clear_registry_for_tests, register
from orbital_game.sampling.serialize import (
    serializable_from_primitive,
    serializable_to_primitive,
)


class _ToyKey(StrEnum):
    GAUSSIAN = "toy_gaussian"


@register(_ToyKey.GAUSSIAN)
@dataclass(frozen=True)
class _ToyGaussian:
    mean: jnp.ndarray
    sigma: float
    label: str = "default"


class _ScopeEnum(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class _EnumKey(StrEnum):
    HAS_ENUM = "has_enum"


@register(_EnumKey.HAS_ENUM)
@dataclass(frozen=True)
class _HasEnum:
    label: str
    scope: _ScopeEnum


def setup_module(_module):
    """Re-register the toy classes in case a prior test module cleared the registry."""
    register(_ToyKey.GAUSSIAN)(_ToyGaussian)
    register(_EnumKey.HAS_ENUM)(_HasEnum)


def test_round_trip_with_array_and_scalar_fields():
    obj = _ToyGaussian(mean=jnp.array([1.0, 2.0, 3.0]), sigma=0.5, label="foo")
    primitive = serializable_to_primitive(obj)
    assert primitive["_key"] == "toy_gaussian"
    assert primitive["mean"] == [1.0, 2.0, 3.0]
    assert primitive["sigma"] == 0.5
    assert primitive["label"] == "foo"

    restored = serializable_from_primitive(primitive, _ToyKey)
    assert isinstance(restored, _ToyGaussian)
    assert jnp.allclose(restored.mean, obj.mean)
    assert restored.sigma == obj.sigma
    assert restored.label == obj.label


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="No callable"):
        serializable_from_primitive({"_key": "unknown", "mean": [], "sigma": 0.0}, _ToyKey)


def test_unregistered_class_raises_on_serialize():
    @dataclass(frozen=True)
    class _NotRegistered:
        x: float

    with pytest.raises(KeyError, match="not registered"):
        serializable_to_primitive(_NotRegistered(x=1.0))


def test_enum_field_round_trip():
    obj = _HasEnum(label="foo", scope=_ScopeEnum.RIGHT)
    primitive = serializable_to_primitive(obj)
    assert primitive["scope"] == "right"
    restored = serializable_from_primitive(primitive, _EnumKey)
    assert restored.scope is _ScopeEnum.RIGHT
    assert restored.label == "foo"


def test_missing_required_field_raises_clear_error():
    obj = _ToyGaussian(mean=jnp.array([1.0]), sigma=0.5, label="foo")
    primitive = serializable_to_primitive(obj)
    primitive.pop("mean")
    with pytest.raises(KeyError, match="mean"):
        serializable_from_primitive(primitive, _ToyKey)


def teardown_module(_module):
    _clear_registry_for_tests()
