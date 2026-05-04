"""Tests for enum keys + registry (serialization identity only)."""

from enum import StrEnum

import pytest

from orbital_game import registry as _registry
from orbital_game.registry import (
    ActuatorKey,
    BeliefInitializerKey,
    BeliefUpdaterKey,
    DynamicsKey,
    DynamicsKind,
    Frame,
    MassSamplerKey,
    ObservationFnKey,
    PolicyKey,
    RewardFnKey,
    SideSamplerKey,
    StateComponentKey,
    TerminationFnKey,
    ValidatorKey,
    _clear_registry_for_tests,
    register,
    resolve,
    resolve_class_to_key,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot the registry, clear for the test, then restore.

    Avoids leaving the registry empty after the last test in the module — other
    tests rely on package-level @register decorators that only run once per
    process (Python import cache).
    """
    fwd = dict(_registry._REGISTRY)
    rev = dict(_registry._REVERSE)
    games_fwd = dict(_registry._GAME_REGISTRY)
    games_rev = dict(_registry._GAME_REVERSE)
    _clear_registry_for_tests()
    try:
        yield
    finally:
        _clear_registry_for_tests()
        _registry._REGISTRY.update(fwd)
        _registry._REVERSE.update(rev)
        _registry._GAME_REGISTRY.update(games_fwd)
        _registry._GAME_REVERSE.update(games_rev)


def test_enum_str_roundtrip():
    """str, Enum inheritance: enum.value is a str; Enum(value) reconstructs."""
    key = DynamicsKey.HCW_RTN
    assert isinstance(key.value, str)
    assert DynamicsKey(key.value) is key


def test_register_and_resolve():
    @register(DynamicsKey.HCW_RT, frame=Frame.RT, kind=DynamicsKind.RELATIVE)
    def _dummy_hcw_rt(*args, **kwargs):
        return "ok"

    assert resolve(DynamicsKey.HCW_RT)() == "ok"


def test_resolve_missing_key_raises():
    with pytest.raises(KeyError):
        resolve(DynamicsKey.HCW_RTN)  # not registered yet


def test_each_enum_has_at_least_one_member():
    """Spec sanity — no empty enum was accidentally shipped."""
    for enum_cls in (
        StateComponentKey,
        DynamicsKey,
        ActuatorKey,
        PolicyKey,
        ObservationFnKey,
        RewardFnKey,
        TerminationFnKey,
        SideSamplerKey,
        MassSamplerKey,
        ValidatorKey,
        BeliefInitializerKey,
        BeliefUpdaterKey,
    ):
        assert len(list(enum_cls)) >= 1


class _TestKey(StrEnum):
    ALPHA = "alpha"


def test_resolve_class_to_key_returns_enum_member_and_class():
    @register(_TestKey.ALPHA)
    class Alpha:
        pass

    enum_value, enum_class = resolve_class_to_key(Alpha)
    assert enum_value == "alpha"
    assert enum_class is _TestKey


def test_resolve_class_to_key_raises_for_unregistered():
    class NotRegistered:
        pass

    with pytest.raises(KeyError, match="not registered"):
        resolve_class_to_key(NotRegistered)
