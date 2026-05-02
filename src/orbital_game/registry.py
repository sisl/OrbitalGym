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


class PolicyKey(StrEnum):
    """Role-agnostic policy registry key. Same key serves any side."""

    ZERO_CONTROL = "zero_control"


class ObservationFnKey(StrEnum):
    FULL = "full_observation"
    ONBOARD_GPS = "onboard_gps_observation"
    RANGE_LIMITED = "range_limited_observation"
    COMPOSITE = "composite_observation"


class RewardFnKey(StrEnum):
    DISTANCE_TO_REFERENCE_ORBIT = "distance_to_reference_orbit"
    PURSUIT_EVASION = "pursuit_evasion_reward"
    SUN_BLOCKING = "sun_blocking_reward"
    OBSERVATION_BLOCKING = "observation_blocking_reward"  # added Phase 2 Task 5


class TerminationFnKey(StrEnum):
    MAX_STEPS_OR_BREACH = "max_steps_or_breach"
    PURSUIT_EVASION = "pursuit_evasion_termination"


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
    KF_FROM_TRUTH = "kf_from_truth"
    KF_UNIFORM_DEFAULT = "kf_uniform_default"


class BeliefUpdaterKey(StrEnum):
    KF = "kf"


class GameKey(StrEnum):
    """Game catalog key. NoGame is the default for custom scenarios."""

    NONE = "no_game"
    LADY_BANDIT_GUARD = "lady_bandit_guard"
    PURSUIT_EVASION = "pursuit_evasion"
    SUN_BLOCKING = "sun_blocking"
    OBSERVATION_BLOCKING = "observation_blocking"


_REGISTRY: dict[Enum, Callable] = {}
_REVERSE: dict[Callable, tuple[str, type[Enum]]] = {}
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


def resolve_class_to_key(cls: Callable) -> tuple[str, type[Enum]]:
    """Reverse lookup: registered class/function -> (enum_value, enum_class).

    Raises KeyError if not registered. The reverse map accepts any callable
    (functions for stateless registrations like dynamics step fns; classes
    for dataclass-based pluggables) since `register` is decorator-bound to
    `Callable`. Round-trip serialization (`sampling/serialize.py`) only
    consults this for dataclass classes, but the storage shape is broader.
    """
    if cls not in _REVERSE:
        name = getattr(cls, "__qualname__", repr(cls))
        raise KeyError(f"Callable {name} is not registered")
    return _REVERSE[cls]


_GAME_REGISTRY: dict[GameKey, type] = {}
_GAME_REVERSE: dict[type, GameKey] = {}


def register_game(key: GameKey) -> Callable[[type], type]:
    """Decorator: @register_game(GameKey.LADY_BANDIT_GUARD) class LadyBanditGuard(Game): ..."""

    def decorator(cls: type) -> type:
        _GAME_REGISTRY[key] = cls
        _GAME_REVERSE[cls] = key
        return cls

    return decorator


def resolve_game(key: GameKey) -> type:
    if key not in _GAME_REGISTRY:
        raise KeyError(f"No game class registered for {key!r}")
    return _GAME_REGISTRY[key]


def resolve_game_class_to_key(cls: type) -> GameKey:
    if cls not in _GAME_REVERSE:
        raise KeyError(f"Game class {cls.__qualname__} is not registered")
    return _GAME_REVERSE[cls]


def _clear_registry_for_tests() -> None:
    """Test-only hook to reset the registry between test cases."""
    _REGISTRY.clear()
    _REVERSE.clear()
    _GAME_REGISTRY.clear()
    _GAME_REVERSE.clear()
