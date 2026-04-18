"""Enum keys + name registry for pluggable callables.

Scope: serialization identity only. OrbitalGameEnv may also be constructed
directly from callables in memory, bypassing the registry. The registry
exists so a persisted ScenarioConfig can be reconstructed in a later session.
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
    CONTINUOUS = "continuous"


class IntruderPolicyKey(StrEnum):
    ZERO_CONTROL = "zero_control_intruder"


class ObservationFnKey(StrEnum):
    FULL = "full_observation"


class RewardFnKey(StrEnum):
    DISTANCE_TO_HVA = "distance_to_hva"


class TerminationFnKey(StrEnum):
    MAX_STEPS_OR_BREACH = "max_steps_or_breach"


class InitialConditionSamplerKey(StrEnum):
    GAUSSIAN_AROUND_NOMINAL = "gaussian_around_nominal"


class BeliefInitializerKey(StrEnum):
    GAUSSIAN_FROM_TRUTH = "gaussian_from_truth"
    GAUSSIAN_UNIFORM_DEFAULT = "gaussian_uniform_default"


class BeliefUpdaterKey(StrEnum):
    GAUSSIAN_KALMAN = "gaussian_kalman"


_REGISTRY: dict[Enum, Callable] = {}
F = TypeVar("F", bound=Callable)


def register(key: Enum) -> Callable[[F], F]:
    """Decorator: @register(DynamicsKey.HCW_RTN) def hcw_rtn_step(...): ..."""

    def decorator(fn: F) -> F:
        _REGISTRY[key] = fn
        return fn

    return decorator


def resolve(key: Enum) -> Callable:
    """Look up the callable registered against `key`. Raises KeyError if missing."""
    if key not in _REGISTRY:
        raise KeyError(f"No callable registered for {key!r}")
    return _REGISTRY[key]


def _clear_registry_for_tests() -> None:
    """Test-only hook to reset the registry between test cases."""
    _REGISTRY.clear()
