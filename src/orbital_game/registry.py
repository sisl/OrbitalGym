"""Enum keys + name registry for pluggable callables.

Scope: serialization identity only. OrbitalGameEnv may also be constructed
directly from callables in memory, bypassing the registry. The registry
exists so a persisted ScenarioConfig can be reconstructed in a later session.

Two indices are maintained:
  - forward (enum_member -> callable/class) used at deserialize time
  - reverse (class -> (enum_value, enum_class)) used at serialize time
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, StrEnum
from typing import TypeVar


class StateComponentKey(StrEnum):
    RT = "rt"
    RTN = "rtn"
    MASS = "mass"
    POWER = "power"
    ATTITUDE = "attitude"
    BODY_RATES = "body_rates"


class DynamicsKey(StrEnum):
    HCW_RT = "hcw_rt"
    HCW_RTN = "hcw_rtn"


class ActuatorKey(StrEnum):
    IMPULSIVE = "impulsive"


class BanditPolicyKey(StrEnum):
    ZERO_CONTROL = "zero_control_bandit"


class ObservationFnKey(StrEnum):
    FULL = "full_observation"


class RewardFnKey(StrEnum):
    DISTANCE_TO_REFERENCE_ORBIT = "distance_to_reference_orbit"


class TerminationFnKey(StrEnum):
    MAX_STEPS_OR_BREACH = "max_steps_or_breach"


class SideSamplerKey(StrEnum):
    RELATIVE_KEPLERIAN = "relative_keplerian"
    RELATIVE_ELLIPSE = "relative_ellipse"


class MassSamplerKey(StrEnum):
    CONSTANT = "constant_mass"
    UNIFORM = "uniform_mass"


class ValidatorKey(StrEnum):
    MIN_SEPARATION = "min_separation"
    MAX_RANGE = "max_range"


class BeliefInitializerKey(StrEnum):
    GAUSSIAN_FROM_TRUTH = "gaussian_from_truth"
    GAUSSIAN_UNIFORM_DEFAULT = "gaussian_uniform_default"


class BeliefUpdaterKey(StrEnum):
    GAUSSIAN_KALMAN = "gaussian_kalman"


_REGISTRY: dict[Enum, Callable] = {}
_REVERSE: dict[type, tuple[str, type[Enum]]] = {}
F = TypeVar("F", bound=Callable)


def register(key: Enum) -> Callable[[F], F]:
    """Decorator: @register(DynamicsKey.HCW_RTN) def hcw_rtn_step(...): ...

    Populates both the forward (key -> callable) and reverse
    (class -> (enum_value, enum_class)) maps. The reverse map is used by the
    sampling/validator generic serializer.
    """

    def decorator(fn: F) -> F:
        _REGISTRY[key] = fn
        _REVERSE[fn] = (key.value, type(key))
        return fn

    return decorator


def resolve(key: Enum) -> Callable:
    """Look up the callable registered against `key`. Raises KeyError if missing."""
    if key not in _REGISTRY:
        raise KeyError(f"No callable registered for {key!r}")
    return _REGISTRY[key]


def resolve_class_to_key(cls: type) -> tuple[str, type[Enum]]:
    """Reverse lookup: class -> (enum_value, enum_class). Raises KeyError if not registered."""
    if cls not in _REVERSE:
        raise KeyError(f"Class {cls.__qualname__} is not registered")
    return _REVERSE[cls]


def _clear_registry_for_tests() -> None:
    """Test-only hook to reset the registry between test cases."""
    _REGISTRY.clear()
    _REVERSE.clear()
