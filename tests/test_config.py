"""Tests for ScenarioConfig and VehicleParamsSpec."""

import jax.numpy as jnp
import pytest

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import (
    DynamicsKey,
    StateComponentKey,
)
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec


def make_config(**overrides):
    base = dict(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=VehicleParamsSpec(
            dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0
        ),
        bandit_params=VehicleParamsSpec(
            dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=2.0
        ),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=0.0,
                phase_rad=0.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=0.0,
                phase_rad=0.0,
            ),
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )
    base.update(overrides)
    return ScenarioConfig(**base)


def test_max_steps_derived_from_horizon_and_dt():
    cfg = make_config(dt=10.0, max_horizon_s=2000.0)
    assert cfg.max_steps == 200


def test_max_steps_ceils_on_non_integer():
    cfg = make_config(dt=3.0, max_horizon_s=10.0)
    assert cfg.max_steps == 4


def test_json_roundtrip_preserves_enum_identity():
    cfg = make_config()
    s = cfg.to_json()
    cfg2 = ScenarioConfig.from_json(s)
    assert cfg2.guard_components == cfg.guard_components
    assert cfg2.dt == cfg.dt
    assert cfg2.truth_dynamics is DynamicsKey.HCW_RTN


def test_post_init_rejects_zero_dt():
    with pytest.raises(ValueError):
        make_config(dt=0.0)


def test_post_init_rejects_horizon_smaller_than_dt():
    with pytest.raises(ValueError):
        make_config(dt=10.0, max_horizon_s=5.0)


def test_config_round_trips_with_relative_ellipse_ic():
    cfg = ScenarioConfig(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7546.05, 0.0]),
        ),
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        bandit_params=VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=2.0),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=100.0,
                phase_rad=0.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=100.0,
                phase_rad=jnp.pi,
            ),
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )
    s = cfg.to_json()
    cfg2 = ScenarioConfig.from_json(s)
    assert cfg2.n_guards == 1
    assert isinstance(cfg2.ic_sampler.guard_sampler, RelativeEllipse)
    assert isinstance(cfg2.ic_sampler.guard_sampler.mass_sampler, ConstantMass)
    assert cfg2.ic_sampler.guard_sampler.mass_sampler.propellant_mass_kg == 10.0


def test_vehicle_params_spec_no_longer_has_propellant_mass_kg():
    spec = VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0)
    assert not hasattr(spec, "propellant_mass_kg")


def test_default_typed_pluggables_are_instances():
    """typed-instance fields are populated as object instances, not enum keys."""
    from orbital_game.observations.reference import FullObservation
    from orbital_game.rewards.reference import DistanceToReferenceOrbit
    from orbital_game.termination.reference import MaxStepsOrBreach
    cfg = make_config()
    assert isinstance(cfg.guard_observation_fn, FullObservation)
    assert isinstance(cfg.bandit_observation_fn, FullObservation)
    assert isinstance(cfg.reward_fn, DistanceToReferenceOrbit)
    assert isinstance(cfg.termination_fn, MaxStepsOrBreach)


def test_default_scripted_policies_are_zero_control():
    from orbital_game.policies.library import ZeroControl
    cfg = make_config()
    assert isinstance(cfg.guard_scripted_policy, ZeroControl)
    assert isinstance(cfg.bandit_scripted_policy, ZeroControl)


def test_default_controlled_side_is_guard():
    from orbital_game.env.types import Side
    cfg = make_config()
    assert cfg.controlled_side is Side.GUARD


def test_roundtrip_typed_pluggables():
    from examples.reference_scenario import build_config
    cfg = build_config()
    s = cfg.to_json()
    cfg2 = ScenarioConfig.from_json(s)
    assert type(cfg.reward_fn) is type(cfg2.reward_fn)
    assert type(cfg.guard_observation_fn) is type(cfg2.guard_observation_fn)
    assert cfg.controlled_side == cfg2.controlled_side
