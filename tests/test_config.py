"""Tests for ScenarioConfig and VehicleParamsSpec."""

import jax.numpy as jnp
import pytest

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.hva import HVAState
from orbital_game.registry import (
    DynamicsKey,
    StateComponentKey,
)
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec


def make_config(**overrides):
    base = dict(
        n_defenders=1,
        n_intruders=1,
        epoch_mjd_utc=60067.0,
        hva=HVAState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        defender_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        intruder_components=(StateComponentKey.RTN,),
        defender_params=VehicleParamsSpec(
            dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0
        ),
        intruder_params=VehicleParamsSpec(
            dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=2.0
        ),
        ic_sampler=ICSpec(
            defender_sampler=RelativeEllipse(
                radial_ellipse_m=0.0,
                phase_rad=0.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            intruder_sampler=RelativeEllipse(
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
    assert cfg2.defender_components == cfg.defender_components
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
        n_defenders=1,
        n_intruders=1,
        epoch_mjd_utc=60067.0,
        hva=HVAState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7546.05, 0.0]),
        ),
        defender_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        intruder_components=(StateComponentKey.RTN,),
        defender_params=VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        intruder_params=VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=2.0),
        ic_sampler=ICSpec(
            defender_sampler=RelativeEllipse(
                radial_ellipse_m=100.0,
                phase_rad=0.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            intruder_sampler=RelativeEllipse(
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
    assert cfg2.n_defenders == 1
    assert isinstance(cfg2.ic_sampler.defender_sampler, RelativeEllipse)
    assert isinstance(cfg2.ic_sampler.defender_sampler.mass_sampler, ConstantMass)
    assert cfg2.ic_sampler.defender_sampler.mass_sampler.propellant_mass_kg == 10.0


def test_vehicle_params_spec_no_longer_has_propellant_mass_kg():
    spec = VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0)
    assert not hasattr(spec, "propellant_mass_kg")
