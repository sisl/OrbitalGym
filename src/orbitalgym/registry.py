"""Enum keys + name registry for swappable callables.

Scope: serialization identity only. OrbitalGymEnv may also be constructed
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


class Frame(StrEnum):
    """Spatial frame in which a state or action is expressed."""

    RT = "rt"  # 2D in-plane radial + along-track
    RTN = "rtn"  # 3D radial-tangential-normal (rotating, ref-orbit local)
    ECI = "eci"  # Earth-Centered Inertial (absolute)

    @property
    def dim(self) -> int:
        """Dimensionality of a position/velocity vector in this frame."""
        return 2 if self is Frame.RT else 3


class DynamicsKind(StrEnum):
    """Whether a dynamics propagates absolute or relative state."""

    RELATIVE = "relative"  # rotating-frame propagation (HCW, etc.)
    ABSOLUTE = "absolute"  # inertial-frame propagation (Keplerian, J2, ...)


class StateComponentKey(StrEnum):
    RT = "rt"
    RTN = "rtn"
    ECI = "eci"
    MASS = "mass"
    POWER = "power"
    ATTITUDE = "attitude"
    BODY_RATES = "body_rates"
    APPLIED_DV = "applied_dv"  # transient — Δv applied this step
    APPLIED_TORQUE = "applied_torque"  # transient — torque applied this step


class DynamicsKey(StrEnum):
    HCW_RT = "hcw_rt"
    HCW_RTN = "hcw_rtn"
    KEPLERIAN_ECI = "keplerian_eci"
    J2_ECI = "j2_eci"
    # Configurable astrojax-backed orbit dynamics. Unlike the others (registered
    # functions), this key resolves to a *class* (`AstrojaxOrbitDynamics`) which
    # callers instantiate with a `ForceModelConfig`. The four ScenarioConfig
    # role fields accept either a `DynamicsKey` (registered function) or a
    # typed instance carrying its own `frame` / `kind` attributes.
    ASTROJAX_ORBIT = "astrojax_orbit"


class AttitudeDynamicsKey(StrEnum):
    """Attitude-layer dynamics.

    Separate from DynamicsKey so the spatial-dynamics validator is unaware
    of attitude propagators.
    """

    RIGID_BODY = "rigid_body_attitude"
    KINEMATIC = "kinematic_attitude"


class ActionComponentKey(StrEnum):
    """Composable per-side action-pytree component."""

    IMPULSIVE_MANEUVER = "impulsive_maneuver"  # batteries-included impulsive Δv + propellant
    COMMUNICATE = "communicate"  # broadcast to other agents (Dec-POMDP comms)
    # body-frame torque (N·m) producer; reaction-wheel saturation modeled in dynamics
    ATTITUDE_CONTROL = "attitude_control"
    POINT_AT = "point_at"  # slew-limited kinematic pointing


class PolicyKey(StrEnum):
    """Role-agnostic policy registry key. Same key serves any side."""

    ZERO_CONTROL = "zero_control"
    POINTING = "pointing_policy"


class ObservationFnKey(StrEnum):
    FULL = "full_observation"
    ONBOARD_GPS = "onboard_gps_observation"
    RANGE_LIMITED = "range_limited_observation"
    COMPOSITE = "composite_observation"
    COMMS_LEAK = "comms_leak_observation"
    CONICAL = "conical_observation"  # conical field-of-view sensor


class RewardFnKey(StrEnum):
    DISTANCE_TO_REFERENCE_ORBIT = "distance_to_reference_orbit"
    ZERO = "zero_reward"
    PURSUIT_EVASION = "pursuit_evasion_reward"
    SUN_BLOCKING = "sun_blocking_reward"
    OBSERVATION_BLOCKING = "observation_blocking_reward"  # added Phase 2 Task 5
    LBG_WITH_COMMS = "lbg_with_comms_reward"
    LBG_ZERO_SUM = "lbg_zero_sum_reward"
    GET_OFF_MY_LAWN = "get_off_my_lawn_reward"


class TerminationFnKey(StrEnum):
    MAX_STEPS_ONLY = "max_steps_only"
    LBG_EVENTS = "lbg_events_termination"
    PURSUIT_EVASION = "pursuit_evasion_termination"
    GET_OFF_MY_LAWN = "get_off_my_lawn_termination"
    MAX_DISTANCE = "max_distance_termination"
    ANY_OF = "any_of_termination"


class SideSamplerKey(StrEnum):
    RELATIVE_KEPLERIAN = "relative_keplerian"
    RELATIVE_ELLIPSE = "relative_ellipse"


class MassSamplerKey(StrEnum):
    CONSTANT = "constant_mass"
    UNIFORM = "uniform_mass"


class AttitudeSamplerKey(StrEnum):
    IDENTITY = "identity_attitude"
    FIXED = "fixed_attitude"
    UNIFORM_QUAT = "uniform_attitude"
    UNIFORM_RATES = "uniform_body_rates"
    UNIFORM_QUAT_AND_RATES = "uniform_attitude_and_rates"


class ValidatorKey(StrEnum):
    MIN_SEPARATION = "min_separation"
    MAX_RANGE = "max_range"


class BeliefInitializerKey(StrEnum):
    KF_FROM_TRUTH = "kf_from_truth"
    KF_UNIFORM_DEFAULT = "kf_uniform_default"
    EKF_FROM_TRUTH = "ekf_from_truth"
    EKF_UNIFORM_DEFAULT = "ekf_uniform_default"
    PF_FROM_TRUTH = "pf_from_truth"
    PF_RING = "pf_ring"


class BeliefUpdaterKey(StrEnum):
    KF = "kf"
    EKF = "ekf"
    PF = "pf"


class BeliefSyncKey(StrEnum):
    KF_TEAM_FUSION = "kf_team_fusion"
    EKF_TEAM_FUSION = "ekf_team_fusion"
    PF_TEAM_FUSION = "pf_team_fusion"


class LinkKey(StrEnum):
    """Team communication link predicate."""

    ALWAYS = "always_linked"
    GROUND_NETWORK = "ground_network_link"
    POINTING_CONE = "pointing_cone_link"


class GameKey(StrEnum):
    """Game catalog key. NoGame is the default for custom scenarios."""

    NONE = "no_game"
    LADY_BANDIT_GUARD = "lady_bandit_guard"
    PURSUIT_EVASION = "pursuit_evasion"
    SUN_BLOCKING = "sun_blocking"
    OBSERVATION_BLOCKING = "observation_blocking"
    GET_OFF_MY_LAWN = "get_off_my_lawn"


_REGISTRY: dict[Enum, Callable] = {}
_REVERSE: dict[Callable, tuple[str, type[Enum]]] = {}
F = TypeVar("F", bound=Callable)


def register(
    key: Enum,
    *,
    frame: Frame | None = None,
    kind: DynamicsKind | None = None,
) -> Callable[[F], F]:
    """Decorator: @register(DynamicsKey.HCW_RTN, frame=Frame.RTN, kind=DynamicsKind.RELATIVE)

    Populates both the forward (key -> callable) and reverse
    (class -> (enum_value, enum_class)) maps. The reverse map is used by the
    sampling/validator generic serializer.

    Optional `frame` and `kind` kwargs attach metadata to the registered callable;
    used by the dynamics validator to check role/component compatibility. Non-dynamics
    registrations omit them.

    DynamicsKey registrations REQUIRE both ``frame`` and ``kind`` — they describe
    a dynamics, and downstream code (dynamics validator, reference-orbit kind check)
    depends on those attributes being populated. Other key types (PolicyKey,
    ObservationFnKey, RewardFnKey, BeliefInitializerKey, BeliefUpdaterKey, etc.)
    accept ``frame=None, kind=None`` since they don't represent dynamics.
    """
    if isinstance(key, DynamicsKey):
        if frame is None:
            raise ValueError(f"register({key!r}): DynamicsKey registrations require frame=Frame.X")
        if kind is None:
            raise ValueError(
                f"register({key!r}): DynamicsKey registrations require kind=DynamicsKind.Y"
            )

    def decorator(fn: F) -> F:
        _REGISTRY[key] = fn
        _REVERSE[fn] = (key.value, type(key))
        if frame is not None:
            fn.frame = frame  # type: ignore[attr-defined]
        if kind is not None:
            fn.kind = kind  # type: ignore[attr-defined]
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
    for dataclass-based components) since `register` is decorator-bound to
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
