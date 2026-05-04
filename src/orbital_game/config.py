"""ScenarioConfig — the single source of truth for a scenario.

Component references that have per-instance knobs are held as typed instances
(set via __post_init__ defaults or passed directly by the caller). Components
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
    DynamicsKind,
    Frame,
    ObservationFnKey,
    PolicyKey,
    RewardFnKey,
    StateComponentKey,
    TerminationFnKey,
    resolve,
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

    # Dynamics roles. Each accepts either a ``DynamicsKey`` (registered step
    # function — pre-baked offering with no per-instance knobs) or a
    # typed-instance dynamics (e.g. ``AstrojaxOrbitDynamics(force_model=...)``)
    # carrying its own ``frame`` / ``kind`` class attributes. Typed annotation
    # is ``DynamicsKey | Any`` to mirror the ``Any`` pattern used for
    # observation/reward fns; the validator/resolution paths handle both shapes
    # via ``_resolve_dynamics_callable``.
    truth_dynamics: Any = DynamicsKey.HCW_RTN
    policy_dynamics: Any = DynamicsKey.HCW_RTN
    # Optional user-supplied belief-update dynamics. Defaults to policy_dynamics
    # via belief_dynamics_resolved when None. Users should read the resolved
    # field, not this input field.
    belief_dynamics: Any = None
    belief_dynamics_resolved: Any = DynamicsKey.HCW_RTN
    # Optional user-supplied reference-orbit propagator (must be ABSOLUTE-kind).
    # Resolved in __post_init__ to `reference_orbit_dynamics_resolved`; users
    # should read the resolved field, not this input field.
    reference_orbit_dynamics: Any = None
    reference_orbit_dynamics_resolved: Any = DynamicsKey.KEPLERIAN_ECI
    guard_actuator: ActuatorKey = ActuatorKey.IMPULSIVE
    bandit_actuator: ActuatorKey = ActuatorKey.IMPULSIVE

    # Frame in which actuator-emitted Δv is expressed. ``env.step()`` rotates
    # the actuator output from this frame into the truth-dynamics frame using
    # ``convert_action`` before applying it. When left as ``None`` (the
    # default), it resolves in __post_init__ to the truth_dynamics frame —
    # this is the no-op-conversion case (action arrives already in the truth
    # frame). Users wiring an RTN policy against an ECI truth, etc., set this
    # explicitly to the policy's emission frame.
    action_frame: Frame | None = None

    # Derived: per-side components auto-extended with views for every frame
    # consumed by the four resolved dynamics roles plus Frame.RTN (viz). Computed
    # in __post_init__; the canonical input to _COMP_LOOKUP for state assembly.
    guard_components_extended: tuple[StateComponentKey, ...] = ()
    bandit_components_extended: tuple[StateComponentKey, ...] = ()

    # Components that hold typed instances. Defaults via __post_init__ to avoid
    # mutable-default issues and circular imports.
    guard_observation_fn: Any = None  # ObservationFn
    bandit_observation_fn: Any = None
    reward_fn: Any = None  # RewardFn
    termination_fn: Any = None  # TerminationFn

    guard_belief_initializer: Any = None
    guard_belief_updater: Any = None
    bandit_belief_initializer: Any = None
    bandit_belief_updater: Any = None

    # Asymmetric play. `controlled_side` accepts a Side enum or its string value;
    # __post_init__ coerces to Side. Declared as Any to avoid a circular import
    # (config.py is loaded before env.types when orbital_game.__init__ runs).
    controlled_side: Any = "guard"  # Side
    guard_policy: Any = None  # Policy
    bandit_policy: Any = None

    # Game catalog (typed knob bundle; default NoGame for custom scenarios)
    game: Any = None  # populated in __post_init__

    # State layout — derived from n_guards/n_bandits/components. Built once
    # in __post_init__ so custom observation functions can read it via
    # `cfg.layout` without reaching into the default `FullObservation`.
    layout: Any = None  # StateLayout

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

        # The spatial state component must match the truth dynamics frame. Truth
        # is the authoritative role; policy/belief frames need not be in the
        # user-supplied components — they are added by the auto-extension below
        # (computed after role resolution).
        _validate_components_match_dynamics(
            "guard", self.guard_components, "truth_dynamics", self.truth_dynamics
        )
        _validate_components_match_dynamics(
            "bandit", self.bandit_components, "truth_dynamics", self.truth_dynamics
        )

        # Resolve action_frame default to the truth-dynamics frame (no-op
        # conversion is the natural default). The truth_dynamics frame was
        # already validated by _validate_components_match_dynamics above, so
        # truth_frame is guaranteed to be a registered spatial Frame.
        truth_fn = _resolve_dynamics_callable(self.truth_dynamics)
        truth_frame: Frame = truth_fn.frame
        if self.action_frame is None:
            object.__setattr__(self, "action_frame", truth_frame)
        else:
            # action_frame must be one of the registered spatial frames.
            if self.action_frame not in (Frame.RT, Frame.RTN, Frame.ECI):
                raise ValueError(f"action_frame must be RT/RTN/ECI; got {self.action_frame}")
            # Reject explicit lossy conversions: rotating an RTN/ECI action
            # into an RT truth frame drops the cross-track component, which
            # silently discards user intent. The resolved-from-None case
            # trivially matches truth and never hits this branch.
            if truth_frame is Frame.RT and self.action_frame in (Frame.RTN, Frame.ECI):
                raise ValueError(
                    f"action_frame={self.action_frame.value} -> "
                    f"truth_frame={truth_frame.value} is a lossy conversion: "
                    f"the cross-track component of an {self.action_frame.value} "
                    f"Δv would be silently dropped. Use an RT-emitting policy "
                    f"or choose a 3D truth frame (RTN/ECI)."
                )

        # Resolve belief_dynamics: defaults to policy_dynamics when None.
        belief_resolved = (
            self.belief_dynamics if self.belief_dynamics is not None else self.policy_dynamics
        )
        object.__setattr__(self, "belief_dynamics_resolved", belief_resolved)

        # Resolve reference_orbit_dynamics. The reference orbit is propagated in
        # ECI (absolute), so it must be an ABSOLUTE-kind dynamics. If the user
        # didn't supply one: default to truth_dynamics if it's absolute, else
        # KEPLERIAN_ECI. If the user did supply one: validate kind=ABSOLUTE.
        if self.reference_orbit_dynamics is None:
            truth_fn = _resolve_dynamics_callable(self.truth_dynamics)
            if getattr(truth_fn, "kind", None) is DynamicsKind.ABSOLUTE:
                resolved_ref = self.truth_dynamics
            else:
                resolved_ref = DynamicsKey.KEPLERIAN_ECI
        else:
            ref_fn = _resolve_dynamics_callable(self.reference_orbit_dynamics)
            if getattr(ref_fn, "kind", None) is not DynamicsKind.ABSOLUTE:
                raise ValueError(
                    f"reference_orbit_dynamics={_dynamics_label(self.reference_orbit_dynamics)} "
                    f"must be ABSOLUTE; got kind={getattr(ref_fn, 'kind', None)}"
                )
            resolved_ref = self.reference_orbit_dynamics
        object.__setattr__(self, "reference_orbit_dynamics_resolved", resolved_ref)

        # Coerce controlled_side string value to Side enum (deferred import to
        # avoid circular dependency at module load time).
        from orbital_game.env.types import Side as _Side

        if not isinstance(self.controlled_side, _Side):
            object.__setattr__(self, "controlled_side", _Side(self.controlled_side))

        # Auto-extend per-side components with derived-frame views. The three
        # per-side resolved roles (truth, policy, belief) plus Frame.RTN for viz
        # drive which frames must materialize on the per-side state. The
        # reference-orbit propagator runs on EnvState.reference_orbit (not on
        # the per-side pytree), so its frame is not consulted here. The
        # user-supplied components stay at the head of the tuple in their
        # original order; only new components are appended.
        needed_frames: set[Frame] = set()
        for _role_field, _dyn_spec in (
            ("truth_dynamics", self.truth_dynamics),
            ("policy_dynamics", self.policy_dynamics),
            ("belief_dynamics_resolved", belief_resolved),
        ):
            if _dyn_spec is None:
                continue
            _fn = _resolve_dynamics_callable(_dyn_spec)
            _f = getattr(_fn, "frame", None)
            if _f is None:
                raise ValueError(
                    f"{_role_field}={_dynamics_label(_dyn_spec)} has no Frame metadata; "
                    f"register it with @register(..., frame=Frame.X, kind=DynamicsKind.Y) "
                    f"or attach .frame and .kind class attributes"
                )
            needed_frames.add(_f)
        needed_frames.add(Frame.RTN)  # viz requires RTN

        # Iterate Frame in declaration order for deterministic component ordering.
        ordered_needed_frames = tuple(f for f in Frame if f in needed_frames)

        # RT and RTN are mutually exclusive on the per-side state — RT is a 2D
        # in-plane projection of RTN. If the user picked RT, do not auto-add
        # RTN (viz handles 2D scenarios via the RT field directly).
        def _extend(
            side_components: tuple[StateComponentKey, ...],
        ) -> tuple[StateComponentKey, ...]:
            out = list(side_components)
            user_has_rt = StateComponentKey.RT in side_components
            for _f in ordered_needed_frames:
                _comp = _FRAME_TO_COMPONENT.get(_f)
                if _comp is None or _comp in out:
                    continue
                if user_has_rt and _comp is StateComponentKey.RTN:
                    continue
                out.append(_comp)
            return tuple(out)

        object.__setattr__(self, "guard_components_extended", _extend(self.guard_components))
        object.__setattr__(self, "bandit_components_extended", _extend(self.bandit_components))

        # Build the canonical StateLayout once. It is derived purely from
        # n_guards/n_bandits/components, so it's always the same regardless
        # of which obs/reward/policy the user wires in. Custom observation
        # functions read this via cfg.layout. State assembly uses the extended
        # tuples so all derived-frame views are present on the pytree.
        if self.layout is None:
            from orbital_game.env.core import _COMP_LOOKUP
            from orbital_game.state.assemble import build_state_class
            from orbital_game.state.layout import StateLayout

            guard_comps = [_COMP_LOOKUP[k] for k in self.guard_components_extended]
            bandit_comps = [_COMP_LOOKUP[k] for k in self.bandit_components_extended]
            guard_cls = build_state_class(guard_comps, self.n_guards, "GuardState")
            bandit_cls = build_state_class(bandit_comps, self.n_bandits, "BanditState")
            object.__setattr__(
                self,
                "layout",
                StateLayout.build(
                    guard_state_cls=guard_cls,
                    bandit_state_cls=bandit_cls,
                    n_guards=self.n_guards,
                    n_bandits=self.n_bandits,
                ),
            )

        # Defaults for typed-instance components. Local imports to avoid cycles.
        if self.guard_observation_fn is None:
            from orbital_game.observations.reference import FullObservation

            object.__setattr__(self, "guard_observation_fn", FullObservation(layout=self.layout))
        if self.bandit_observation_fn is None:
            from orbital_game.observations.reference import FullObservation

            object.__setattr__(self, "bandit_observation_fn", FullObservation(layout=self.layout))
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
        if self.guard_policy is None:
            from orbital_game.policies import ZeroControl

            object.__setattr__(self, "guard_policy", ZeroControl())
        if self.bandit_policy is None:
            from orbital_game.policies import ZeroControl

            object.__setattr__(self, "bandit_policy", ZeroControl())

        # Default game = NoGame
        if self.game is None:
            object.__setattr__(self, "game", NoGame())

    def to_json(self) -> str:
        return json.dumps(_config_to_primitive(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> ScenarioConfig:
        raw = json.loads(s)
        return _primitive_to_config(raw, cls)


_FRAME_TO_COMPONENT: dict[Frame, StateComponentKey] = {
    Frame.RT: StateComponentKey.RT,
    Frame.RTN: StateComponentKey.RTN,
    Frame.ECI: StateComponentKey.ECI,
}


def _resolve_dynamics_callable(spec: Any) -> Any:
    """Return the underlying callable for a dynamics-role spec.

    ``spec`` may be a ``DynamicsKey`` (registered step function) or a
    typed-instance dynamics (e.g. ``AstrojaxOrbitDynamics``) carrying its
    own ``frame`` / ``kind`` class attributes.
    """
    if isinstance(spec, DynamicsKey):
        return resolve(spec)
    return spec


def _dynamics_label(spec: Any) -> str:
    """Stable human-readable label for a dynamics spec used in error messages."""
    if isinstance(spec, DynamicsKey):
        return spec.value
    return type(spec).__name__


def _validate_components_match_dynamics(
    side: str,
    components: tuple[StateComponentKey, ...],
    dynamics_field: str,
    dynamics: Any,
) -> None:
    """Validate that ``components`` includes the component required by ``dynamics``'s frame.

    NOTE: This validator is intentionally only run against ``truth_dynamics``
    (the user-supplied per-side components must directly cover the truth
    frame). ``policy_dynamics`` and ``belief_dynamics`` rely on the ``_extend``
    auto-extension in ``__post_init__`` to add any frames they need as derived
    views, so they do NOT go through this validator. Do not change that —
    auto-extension users would lose mixed-frame support.
    """
    fn = _resolve_dynamics_callable(dynamics)
    frame: Frame | None = getattr(fn, "frame", None)
    if frame is None:
        raise ValueError(
            f"{dynamics_field}={_dynamics_label(dynamics)} has no Frame metadata; "
            f"register it with @register(..., frame=Frame.X, kind=DynamicsKind.Y) "
            f"or attach class-level frame/kind attributes"
        )
    required = _FRAME_TO_COMPONENT[frame]
    got = [c.value for c in components]
    if required not in components:
        raise ValueError(
            f"{dynamics_field}={_dynamics_label(dynamics)} (frame={frame.value}) requires "
            f"{side}_components to include {required.value!r}; got {got}"
        )
    # Forbid only the *other* registered relative-frame component to preserve the
    # existing "RT and RTN are mutually exclusive" error message. ECI is allowed
    # to coexist with RTN (mixed-frame scenarios — Phase 5 introduces this).
    forbidden_pairs: dict[StateComponentKey, StateComponentKey] = {
        StateComponentKey.RT: StateComponentKey.RTN,
        StateComponentKey.RTN: StateComponentKey.RT,
    }
    forbidden = forbidden_pairs.get(required)
    if forbidden is not None and forbidden in components:
        raise ValueError(
            f"{dynamics_field}={_dynamics_label(dynamics)} is incompatible with "
            f"{forbidden.value!r} in {side}_components; got {got}"
        )


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
    "guard_policy": PolicyKey,
    "bandit_policy": PolicyKey,
}

# Typed-instance fields whose instances contain config-derived non-serializable
# sub-objects (e.g. StateLayout). These are stored as key-only dicts in JSON;
# deserialization leaves them absent from kwargs so __post_init__ re-creates
# them with the correct layout derived from the config's components/sizes.
_KEY_ONLY_TYPED_FIELDS: frozenset[str] = frozenset(
    {
        "guard_observation_fn",
        "bandit_observation_fn",
    }
)

# Fields that are plain enum members (not typed-instance components).
_ENUM_FIELDS: dict[str, type[Enum]] = {
    "guard_components": StateComponentKey,
    "bandit_components": StateComponentKey,
    "truth_dynamics": DynamicsKey,
    "policy_dynamics": DynamicsKey,
    "belief_dynamics": DynamicsKey,
    "reference_orbit_dynamics": DynamicsKey,
    "guard_actuator": ActuatorKey,
    "bandit_actuator": ActuatorKey,
    "action_frame": Frame,
}


def _config_to_primitive(cfg: ScenarioConfig) -> dict[str, Any]:
    from orbital_game.sampling.serialize import serializable_to_primitive

    d: dict[str, Any] = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        # Skip derived fields that __post_init__ rebuilds.
        if f.name in (
            "layout",
            "reference_orbit_dynamics_resolved",
            "belief_dynamics_resolved",
            "guard_components_extended",
            "bandit_components_extended",
        ):
            continue
        # Dynamics role fields accept enum or typed-instance. Enum serializes via
        # .value below; typed-instance serialization is deferred (see spec risk #2).
        if f.name in (
            "truth_dynamics",
            "policy_dynamics",
            "belief_dynamics",
            "reference_orbit_dynamics",
        ):
            if isinstance(v, Enum):
                d[f.name] = v.value
            elif v is None:
                d[f.name] = None
            else:
                raise NotImplementedError(
                    "JSON serialization of typed-instance dynamics is deferred; "
                    "see superpowers/specs/2026-05-04-three-dynamics-roles-and-eci-frame-design.md "
                    "risk #2."
                )
            continue
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
            # Optional enum fields (e.g. reference_orbit_dynamics) may be stored
            # as None — leave absent so __post_init__ resolves the default.
            if v is None:
                continue
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
                kwargs[f.name] = serializable_from_primitive(v, _TYPED_INSTANCE_FIELDS[f.name])
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
