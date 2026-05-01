"""ScenarioConfig — the single source of truth for a scenario.

All pluggable references are typed Enum keys; JSON round-trips via Enum.value.
Scenario horizon is specified in physical time (dt + max_horizon_s); the discrete
step count is a derived property.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any

import jax.numpy as jnp

from orbital_game.hva import HVAState
from orbital_game.registry import (
    ActuatorKey,
    BeliefInitializerKey,
    BeliefUpdaterKey,
    DynamicsKey,
    IntruderPolicyKey,
    ObservationFnKey,
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
    n_defenders: int
    n_intruders: int

    # Scenario time anchor (epoch is scenario-level, not HVA-level)
    epoch_mjd_utc: float

    # HVA kinematic state at epoch
    hva: HVAState

    # Per-side state components
    defender_components: tuple[StateComponentKey, ...]
    intruder_components: tuple[StateComponentKey, ...]

    # Per-side vehicle parameters
    defender_params: VehicleParamsSpec
    intruder_params: VehicleParamsSpec

    # Initial-condition specification. The user composes per-side samplers
    # (RelativeKeplerian, RelativeEllipse, ...), optional validators, and a
    # max_attempts cap; the env runs rejection sampling under jax.lax.while_loop
    # at reset time.
    ic_sampler: ICSpec

    # Environment clock + reproducibility
    dt: float
    max_horizon_s: float
    seed: int

    # Pluggables
    truth_dynamics: DynamicsKey = DynamicsKey.HCW_RTN
    planning_dynamics: DynamicsKey = DynamicsKey.HCW_RTN
    defender_actuator: ActuatorKey = ActuatorKey.IMPULSIVE
    intruder_actuator: ActuatorKey = ActuatorKey.IMPULSIVE
    intruder_policy: IntruderPolicyKey = IntruderPolicyKey.ZERO_CONTROL
    # Separate observation functions per side — defenders and intruders do not
    # share sensor suites in general. Configure independently.
    defender_observation_fn: ObservationFnKey = ObservationFnKey.FULL
    intruder_observation_fn: ObservationFnKey = ObservationFnKey.FULL
    reward_fn: RewardFnKey = RewardFnKey.DISTANCE_TO_HVA
    termination_fn: TerminationFnKey = TerminationFnKey.MAX_STEPS_OR_BREACH
    belief_initializer: BeliefInitializerKey = BeliefInitializerKey.GAUSSIAN_FROM_TRUTH
    belief_updater: BeliefUpdaterKey = BeliefUpdaterKey.GAUSSIAN_KALMAN

    @property
    def max_steps(self) -> int:
        """Derived step count: ceil(max_horizon_s / dt)."""
        return math.ceil(self.max_horizon_s / self.dt)

    def __post_init__(self) -> None:
        if self.n_defenders < 1:
            raise ValueError("n_defenders must be >= 1")
        if self.n_intruders < 1:
            raise ValueError("n_intruders must be >= 1")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.max_horizon_s <= 0:
            raise ValueError("max_horizon_s must be > 0")
        if self.max_horizon_s < self.dt:
            raise ValueError("max_horizon_s must be >= dt")

    def to_json(self) -> str:
        return json.dumps(_config_to_primitive(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> ScenarioConfig:
        raw = json.loads(s)
        return _primitive_to_config(raw, cls)


def _config_to_primitive(cfg: ScenarioConfig) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, Enum):
            d[f.name] = v.value
        elif isinstance(v, tuple) and v and isinstance(v[0], Enum):
            d[f.name] = [x.value for x in v]
        elif isinstance(v, HVAState):
            d[f.name] = {
                "position_eci": [float(x) for x in v.position_eci],
                "velocity_eci": [float(x) for x in v.velocity_eci],
            }
        elif isinstance(v, VehicleParamsSpec):
            d[f.name] = asdict(v)
        elif isinstance(v, ICSpec):
            d[f.name] = _ic_spec_to_primitive(v)
        else:
            d[f.name] = v
    return d


_ENUM_FIELDS: dict[str, type[Enum]] = {
    "defender_components": StateComponentKey,
    "intruder_components": StateComponentKey,
    "truth_dynamics": DynamicsKey,
    "planning_dynamics": DynamicsKey,
    "defender_actuator": ActuatorKey,
    "intruder_actuator": ActuatorKey,
    "intruder_policy": IntruderPolicyKey,
    "defender_observation_fn": ObservationFnKey,
    "intruder_observation_fn": ObservationFnKey,
    "reward_fn": RewardFnKey,
    "termination_fn": TerminationFnKey,
    "belief_initializer": BeliefInitializerKey,
    "belief_updater": BeliefUpdaterKey,
}


def _primitive_to_config(raw: dict[str, Any], cls: type[ScenarioConfig]) -> ScenarioConfig:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        v = raw[f.name]
        if f.name in ("defender_components", "intruder_components"):
            kwargs[f.name] = tuple(StateComponentKey(x) for x in v)
        elif f.name in _ENUM_FIELDS:
            kwargs[f.name] = _ENUM_FIELDS[f.name](v)
        elif f.name == "hva":
            kwargs[f.name] = HVAState(
                position_eci=jnp.asarray(v["position_eci"]),
                velocity_eci=jnp.asarray(v["velocity_eci"]),
            )
        elif f.name in ("defender_params", "intruder_params"):
            kwargs[f.name] = VehicleParamsSpec(**v)
        elif f.name == "ic_sampler":
            kwargs[f.name] = _ic_spec_from_primitive(v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def _ic_spec_to_primitive(spec: ICSpec) -> dict:
    """Serialize an ICSpec for JSON round-trip."""
    from orbital_game.sampling.serialize import serializable_to_primitive
    return {
        "defender_sampler": serializable_to_primitive(spec.defender_sampler),
        "intruder_sampler": serializable_to_primitive(spec.intruder_sampler),
        "validators": [serializable_to_primitive(v) for v in spec.validators],
        "max_attempts": spec.max_attempts,
    }


def _ic_spec_from_primitive(d: dict) -> ICSpec:
    from orbital_game.registry import MassSamplerKey, SideSamplerKey, ValidatorKey
    from orbital_game.sampling.serialize import serializable_from_primitive

    def _hydrate_side_sampler(sd: dict):
        sd = dict(sd)
        if sd.get("mass_sampler") is not None and isinstance(sd["mass_sampler"], dict):
            sd["mass_sampler"] = serializable_from_primitive(sd["mass_sampler"], MassSamplerKey)
        return serializable_from_primitive(sd, SideSamplerKey)

    return ICSpec(
        defender_sampler=_hydrate_side_sampler(d["defender_sampler"]),
        intruder_sampler=_hydrate_side_sampler(d["intruder_sampler"]),
        validators=tuple(serializable_from_primitive(v, ValidatorKey) for v in d["validators"]),
        max_attempts=d["max_attempts"],
    )
