"""ScenarioConfig — the single source of truth for a scenario.

Pluggable references that have per-instance knobs are held as typed instances
(set via __post_init__ defaults or passed directly by the caller). Pluggables
with no per-instance knobs (dynamics, actuators) stay as enum keys for
cheap equality + JSON round-trips.

Scenario horizon is specified in physical time (dt + max_horizon_s); the
discrete step count is a derived property.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any

import jax.numpy as jnp

from orbital_game.games.base import Game, NoGame
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import (
    ActuatorKey,
    BeliefInitializerKey,
    BeliefUpdaterKey,
    DynamicsKey,
    ObservationFnKey,
    PolicyKey,
    RewardFnKey,
    StateComponentKey,
    TerminationFnKey,
)
from orbital_game.sampling.spec import ICSpec


@dataclass(frozen=True)
class VehicleParamsSpec:
    dry_mass_kg: float
    isp_s: float
    max_thrust_n: float


@dataclass(frozen=True)
class ScenarioConfig:
    # Fleet sizing
    n_guards: int
    n_bandits: int

    # Time anchor + reference frame
    epoch_mjd_utc: float
    reference_orbit: ReferenceOrbitState

    # Per-side state composition
    guard_components: tuple[StateComponentKey, ...]
    bandit_components: tuple[StateComponentKey, ...]

    # Per-side vehicle parameters
    guard_params: VehicleParamsSpec
    bandit_params: VehicleParamsSpec

    # IC sampling
    ic_sampler: ICSpec

    # Clock + reproducibility
    dt: float
    max_horizon_s: float
    seed: int

    # Pluggables that stay enum-keyed (no per-instance knobs)
    truth_dynamics: DynamicsKey = DynamicsKey.HCW_RTN
    planning_dynamics: DynamicsKey = DynamicsKey.HCW_RTN
    guard_actuator: ActuatorKey = ActuatorKey.IMPULSIVE
    bandit_actuator: ActuatorKey = ActuatorKey.IMPULSIVE

    # Pluggables that hold typed instances. Defaults via __post_init__ to avoid
    # mutable-default issues and circular imports.
    guard_observation_fn: Any = None     # ObservationFn
    bandit_observation_fn: Any = None
    reward_fn: Any = None                # RewardFn
    termination_fn: Any = None           # TerminationFn

    guard_belief_initializer: Any = None
    guard_belief_updater: Any = None
    bandit_belief_initializer: Any = None
    bandit_belief_updater: Any = None

    # Asymmetric play. `controlled_side` accepts a Side enum or its string value;
    # __post_init__ coerces to Side. Declared as Any to avoid a circular import
    # (config.py is loaded before env.types when orbital_game.__init__ runs).
    controlled_side: Any = "guard"       # Side
    guard_scripted_policy: Any = None    # Policy
    bandit_scripted_policy: Any = None

    # Game catalog (typed knob bundle; default NoGame for custom scenarios)
    game: Any = None  # populated in __post_init__

    @property
    def max_steps(self) -> int:
        """Derived step count: ceil(max_horizon_s / dt)."""
        return math.ceil(self.max_horizon_s / self.dt)

    def __post_init__(self) -> None:
        if self.n_guards < 1:
            raise ValueError("n_guards must be >= 1")
        if self.n_bandits < 1:
            raise ValueError("n_bandits must be >= 1")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.max_horizon_s <= 0:
            raise ValueError("max_horizon_s must be > 0")
        if self.max_horizon_s < self.dt:
            raise ValueError("max_horizon_s must be >= dt")

        # Coerce controlled_side string value to Side enum (deferred import to
        # avoid circular dependency at module load time).
        from orbital_game.env.types import Side as _Side
        if not isinstance(self.controlled_side, _Side):
            object.__setattr__(self, "controlled_side", _Side(self.controlled_side))

        # Defaults for typed-instance pluggables. Local imports to avoid cycles.
        if self.guard_observation_fn is None:
            from orbital_game.env.core import _COMP_LOOKUP
            from orbital_game.observations.reference import FullObservation
            from orbital_game.state.assemble import build_state_class
            from orbital_game.state.layout import StateLayout
            guard_comps = [_COMP_LOOKUP[k] for k in self.guard_components]
            bandit_comps = [_COMP_LOOKUP[k] for k in self.bandit_components]
            guard_cls = build_state_class(guard_comps, self.n_guards, "GuardState")
            bandit_cls = build_state_class(bandit_comps, self.n_bandits, "BanditState")
            layout = StateLayout.build(
                guard_state_cls=guard_cls,
                bandit_state_cls=bandit_cls,
                n_guards=self.n_guards,
                n_bandits=self.n_bandits,
            )
            object.__setattr__(self, "guard_observation_fn", FullObservation(layout=layout))
        if self.bandit_observation_fn is None:
            from orbital_game.observations.reference import FullObservation
            guard_obs_fn = self.guard_observation_fn  # always set by this point
            layout = guard_obs_fn.layout  # pyrefly: ignore[missing-attribute]
            object.__setattr__(self, "bandit_observation_fn", FullObservation(layout=layout))
        if self.reward_fn is None:
            from orbital_game.rewards.reference import DistanceToReferenceOrbit
            object.__setattr__(self, "reward_fn", DistanceToReferenceOrbit())
        if self.termination_fn is None:
            from orbital_game.termination.reference import MaxStepsOrBreach
            object.__setattr__(
                self,
                "termination_fn",
                MaxStepsOrBreach(max_steps=self.max_steps, breach_distance_m=10.0),
            )
        # Belief initializer/updater require layout-dependent args (variance_diag,
        # stm, obs_matrix, etc.) and cannot be meaningfully defaulted here.
        # They remain None until the caller or OrbitalGameEnv sets them.
        if self.guard_scripted_policy is None:
            from orbital_game.policies.library import ZeroControl
            object.__setattr__(self, "guard_scripted_policy", ZeroControl())
        if self.bandit_scripted_policy is None:
            from orbital_game.policies.library import ZeroControl
            object.__setattr__(self, "bandit_scripted_policy", ZeroControl())

        # Default game = NoGame
        if self.game is None:
            object.__setattr__(self, "game", NoGame())

    def to_json(self) -> str:
        return json.dumps(_config_to_primitive(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> ScenarioConfig:
        raw = json.loads(s)
        return _primitive_to_config(raw, cls)


# Fields that are serialized as typed-instance dicts via serializable_to_primitive.
# Maps field name -> the registry enum class to use for key lookup.
_TYPED_INSTANCE_FIELDS: dict[str, type[Enum]] = {
    "guard_observation_fn": ObservationFnKey,
    "bandit_observation_fn": ObservationFnKey,
    "reward_fn": RewardFnKey,
    "termination_fn": TerminationFnKey,
    "guard_belief_initializer": BeliefInitializerKey,
    "guard_belief_updater": BeliefUpdaterKey,
    "bandit_belief_initializer": BeliefInitializerKey,
    "bandit_belief_updater": BeliefUpdaterKey,
    "guard_scripted_policy": PolicyKey,
    "bandit_scripted_policy": PolicyKey,
}

# Typed-instance fields whose instances contain config-derived non-serializable
# sub-objects (e.g. StateLayout). These are stored as key-only dicts in JSON;
# deserialization leaves them absent from kwargs so __post_init__ re-creates
# them with the correct layout derived from the config's components/sizes.
_KEY_ONLY_TYPED_FIELDS: frozenset[str] = frozenset({
    "guard_observation_fn",
    "bandit_observation_fn",
})

# Fields that are plain enum members (not typed-instance pluggables).
_ENUM_FIELDS: dict[str, type[Enum]] = {
    "guard_components": StateComponentKey,
    "bandit_components": StateComponentKey,
    "truth_dynamics": DynamicsKey,
    "planning_dynamics": DynamicsKey,
    "guard_actuator": ActuatorKey,
    "bandit_actuator": ActuatorKey,
}


def _config_to_primitive(cfg: ScenarioConfig) -> dict[str, Any]:
    from orbital_game.sampling.serialize import serializable_to_primitive
    d: dict[str, Any] = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, Enum):
            d[f.name] = v.value
        elif isinstance(v, tuple) and v and isinstance(v[0], Enum):
            d[f.name] = [x.value for x in v]
        elif isinstance(v, ReferenceOrbitState):
            d[f.name] = {
                "position_eci": [float(x) for x in v.position_eci],
                "velocity_eci": [float(x) for x in v.velocity_eci],
            }
        elif isinstance(v, VehicleParamsSpec):
            d[f.name] = asdict(v)
        elif isinstance(v, ICSpec):
            d[f.name] = _ic_spec_to_primitive(v)
        elif f.name in _TYPED_INSTANCE_FIELDS:
            if v is None:
                d[f.name] = None
            elif f.name in _KEY_ONLY_TYPED_FIELDS:
                # Observation fns hold a config-derived StateLayout that is not
                # independently serializable. Store only the registry key; the
                # layout is re-derived from config fields during from_json.
                from orbital_game.registry import resolve_class_to_key
                key_value, _ = resolve_class_to_key(type(v))
                d[f.name] = {"_key": key_value}
            else:
                d[f.name] = serializable_to_primitive(v)
        elif f.name == "game":
            d[f.name] = _game_to_primitive(v)
        else:
            d[f.name] = v
    return d


def _primitive_to_config(raw: dict[str, Any], cls: type[ScenarioConfig]) -> ScenarioConfig:
    from orbital_game.sampling.serialize import serializable_from_primitive
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            # Field absent from JSON — skip and let __post_init__ supply default.
            continue
        v = raw[f.name]
        if f.name in ("guard_components", "bandit_components"):
            kwargs[f.name] = tuple(StateComponentKey(x) for x in v)
        elif f.name in _ENUM_FIELDS:
            kwargs[f.name] = _ENUM_FIELDS[f.name](v)
        elif f.name == "reference_orbit":
            kwargs[f.name] = ReferenceOrbitState(
                position_eci=jnp.asarray(v["position_eci"]),
                velocity_eci=jnp.asarray(v["velocity_eci"]),
            )
        elif f.name in ("guard_params", "bandit_params"):
            kwargs[f.name] = VehicleParamsSpec(**v)
        elif f.name == "ic_sampler":
            kwargs[f.name] = _ic_spec_from_primitive(v)
        elif f.name in _TYPED_INSTANCE_FIELDS:
            if v is None:
                # Leave absent so __post_init__ supplies the default instance.
                pass
            elif f.name in _KEY_ONLY_TYPED_FIELDS:
                # Key-only serialized fields: the stored dict has only '_key'.
                # Leave absent so __post_init__ re-derives from config layout.
                pass
            else:
                kwargs[f.name] = serializable_from_primitive(
                    v, _TYPED_INSTANCE_FIELDS[f.name]
                )
        elif f.name == "controlled_side":
            from orbital_game.env.types import Side as _Side
            kwargs[f.name] = _Side(v)
        elif f.name == "game":
            kwargs[f.name] = _game_from_primitive(v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def _ic_spec_to_primitive(spec: ICSpec) -> dict:
    """Serialize an ICSpec for JSON round-trip."""
    from orbital_game.sampling.serialize import serializable_to_primitive
    return {
        "guard_sampler": serializable_to_primitive(spec.guard_sampler),
        "bandit_sampler": serializable_to_primitive(spec.bandit_sampler),
        "validators": [serializable_to_primitive(v) for v in spec.validators],
        "max_attempts": spec.max_attempts,
    }


def _game_to_primitive(g: Game) -> dict:
    """Serialize a Game subclass instance to a JSON-friendly dict.

    Only init=True fields are included — computed fields (init=False, e.g.
    target_ecef_m on ObservationBlocking) are excluded since they are derived
    from the init fields in __post_init__ and are not JSON-serializable.
    """
    from dataclasses import fields as _fields

    from orbital_game.registry import resolve_game_class_to_key

    key = resolve_game_class_to_key(type(g))
    payload = {f.name: getattr(g, f.name) for f in _fields(g) if f.init}
    return {"_key": key.value, **payload}


def _game_from_primitive(d: dict) -> Game:
    from dataclasses import fields as _fields

    from orbital_game.registry import GameKey, resolve_game
    payload = dict(d)
    key_value = payload.pop("_key")
    key = GameKey(key_value)
    cls = resolve_game(key)
    # Game subclasses with init=False fields (e.g. ObservationBlocking.target_ecef_m
    # in future tasks) need those omitted from kwargs since they're computed in
    # __post_init__.
    init_field_names = {f.name for f in _fields(cls) if f.init}
    kwargs = {k: v for k, v in payload.items() if k in init_field_names}
    return cls(**kwargs)


def _ic_spec_from_primitive(d: dict) -> ICSpec:
    from orbital_game.registry import MassSamplerKey, SideSamplerKey, ValidatorKey
    from orbital_game.sampling.serialize import serializable_from_primitive

    def _hydrate_side_sampler(sd: dict):
        sd = dict(sd)
        if sd.get("mass_sampler") is not None and isinstance(sd["mass_sampler"], dict):
            sd["mass_sampler"] = serializable_from_primitive(sd["mass_sampler"], MassSamplerKey)
        return serializable_from_primitive(sd, SideSamplerKey)

    return ICSpec(
        guard_sampler=_hydrate_side_sampler(d["guard_sampler"]),
        bandit_sampler=_hydrate_side_sampler(d["bandit_sampler"]),
        validators=tuple(serializable_from_primitive(v, ValidatorKey) for v in d["validators"]),
        max_attempts=d["max_attempts"],
    )
