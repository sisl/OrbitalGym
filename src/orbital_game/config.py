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
from orbital_game.sampling.reference import GaussianAroundNominal


@dataclass(frozen=True)
class VehicleParamsSpec:
    dry_mass_kg: float
    propellant_mass_kg: float
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

    # Initial-condition sampler. The user constructs this with their chosen
    # nominal states + noise magnitudes; the env uses it directly at reset.
    # Typed concretely as GaussianAroundNominal for the bootstrap; if more
    # IC samplers land, promote to a union or a structural protocol.
    ic_sampler: GaussianAroundNominal

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
        if self.defender_params.propellant_mass_kg < 0:
            raise ValueError("defender propellant_mass_kg must be >= 0")
        if self.intruder_params.propellant_mass_kg < 0:
            raise ValueError("intruder propellant_mass_kg must be >= 0")

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
        elif isinstance(v, GaussianAroundNominal):
            d[f.name] = {
                "nominal_defender_state": [
                    [float(x) for x in row] for row in v.nominal_defender_state
                ],
                "nominal_intruder_state": [
                    [float(x) for x in row] for row in v.nominal_intruder_state
                ],
                "sigma_pos": float(v.sigma_pos),
                "sigma_vel": float(v.sigma_vel),
            }
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
            kwargs[f.name] = GaussianAroundNominal(
                nominal_defender_state=jnp.asarray(v["nominal_defender_state"]),
                nominal_intruder_state=jnp.asarray(v["nominal_intruder_state"]),
                sigma_pos=v["sigma_pos"],
                sigma_vel=v["sigma_vel"],
            )
        else:
            kwargs[f.name] = v
    return cls(**kwargs)
