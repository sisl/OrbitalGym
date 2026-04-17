"""Tests for enum keys + registry (serialization identity only)."""
import pytest

from orbital_game.registry import (
    ActuatorKey,
    BeliefInitializerKey,
    BeliefUpdaterKey,
    DynamicsKey,
    InitialConditionSamplerKey,
    IntruderPolicyKey,
    ObservationFnKey,
    RewardFnKey,
    StateComponentKey,
    TerminationFnKey,
    _clear_registry_for_tests,
    register,
    resolve,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Clear the registry before and after every test so order-dependent state never leaks."""
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def test_enum_str_roundtrip():
    """str, Enum inheritance: enum.value is a str; Enum(value) reconstructs."""
    key = DynamicsKey.HCW_RTN
    assert isinstance(key.value, str)
    assert DynamicsKey(key.value) is key


def test_register_and_resolve():
    @register(DynamicsKey.HCW_RT)
    def _dummy_hcw_rt(*args, **kwargs):
        return "ok"

    assert resolve(DynamicsKey.HCW_RT)() == "ok"


def test_resolve_missing_key_raises():
    with pytest.raises(KeyError):
        resolve(DynamicsKey.HCW_RTN)  # not registered yet


def test_each_enum_has_at_least_one_member():
    """Spec sanity — no empty enum was accidentally shipped."""
    for enum_cls in (
        StateComponentKey, DynamicsKey, ActuatorKey, IntruderPolicyKey,
        ObservationFnKey, RewardFnKey, TerminationFnKey,
        InitialConditionSamplerKey, BeliefInitializerKey, BeliefUpdaterKey,
    ):
        assert len(list(enum_cls)) >= 1
