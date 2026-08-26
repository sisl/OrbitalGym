"""Construction-time preflight: components that read `action.sides.guard.active`
must be paired with `ActionComponentKey.COMMUNICATE` in `guard_action_components`.

Without this preflight, missing wiring surfaces as a late AttributeError deep
in the env step / observation call (often inside a jit trace), which is hard
to diagnose. The preflight raises a clear ValueError at `OrbitalGymEnv.__init__`
time naming the offending field.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.observations.comms_leak import CommsLeakObservation
from orbitalgym.observations.composite import CompositeObservation
from orbitalgym.observations.reference import FullObservation
from orbitalgym.reference_orbit import ReferenceOrbitState
from orbitalgym.registry import ActionComponentKey, StateComponentKey
from orbitalgym.rewards.lbg_with_comms import LbgWithCommsReward
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec


def _base_cfg(**overrides):
    cfg_kwargs = dict(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=VehicleParamsSpec(100.0, 220.0, 5.0),
        bandit_params=VehicleParamsSpec(50.0, 200.0, 2.0),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=jnp.pi,
                sigma_radial_ellipse_m=10.0,
            ),
            validators=(),
            max_attempts=100,
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )
    cfg_kwargs.update(overrides)
    return ScenarioConfig(**cfg_kwargs)


def test_comms_leak_observation_without_communicate_raises():
    cfg = _base_cfg(
        bandit_observation_fn=CompositeObservation(
            constituents=(FullObservation(layout=None), CommsLeakObservation(layout=None)),
        ),
        # Note: missing COMMUNICATE on guard side.
        guard_action_components=(ActionComponentKey.IMPULSIVE_MANEUVER,),
    )
    with pytest.raises(ValueError, match="COMMUNICATE"):
        OrbitalGymEnv(cfg)


def test_lbg_with_comms_reward_without_communicate_raises():
    cfg = _base_cfg(
        reward_fn=LbgWithCommsReward(),
        guard_action_components=(ActionComponentKey.IMPULSIVE_MANEUVER,),
    )
    with pytest.raises(ValueError, match="COMMUNICATE"):
        OrbitalGymEnv(cfg)


def test_comms_leak_observation_with_communicate_succeeds():
    cfg = _base_cfg(
        bandit_observation_fn=CommsLeakObservation(layout=None),
        guard_action_components=(
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.COMMUNICATE,
        ),
    )
    env = OrbitalGymEnv(cfg)
    # If we got here, the preflight passed.
    assert env is not None
