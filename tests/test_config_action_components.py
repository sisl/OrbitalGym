"""ScenarioConfig accepts per-side action-component tuples and defaults
to (IMPULSIVE_MANEUVER,) for backward parity."""

import jax.numpy as jnp
import pytest

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import ActionComponentKey, StateComponentKey
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec


def _base_kwargs():
    return dict(
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


def test_default_action_components_is_impulsive_maneuver():
    cfg = ScenarioConfig(**_base_kwargs())
    assert cfg.guard_action_components == (ActionComponentKey.IMPULSIVE_MANEUVER,)
    assert cfg.bandit_action_components == (ActionComponentKey.IMPULSIVE_MANEUVER,)


def test_explicit_action_components_accepted():
    cfg = ScenarioConfig(
        **_base_kwargs(),
        guard_action_components=(
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.COMMUNICATE,
        ),
    )
    assert cfg.guard_action_components == (
        ActionComponentKey.IMPULSIVE_MANEUVER,
        ActionComponentKey.COMMUNICATE,
    )
    # Bandit defaults still impulsive-maneuver-only.
    assert cfg.bandit_action_components == (ActionComponentKey.IMPULSIVE_MANEUVER,)


def test_empty_guard_action_components_rejected():
    with pytest.raises(ValueError, match="guard_action_components"):
        ScenarioConfig(**_base_kwargs(), guard_action_components=())


def test_empty_bandit_action_components_rejected():
    with pytest.raises(ValueError, match="bandit_action_components"):
        ScenarioConfig(**_base_kwargs(), bandit_action_components=())


def test_action_components_round_trip_through_json():
    cfg = ScenarioConfig(
        **_base_kwargs(),
        guard_action_components=(
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.COMMUNICATE,
        ),
    )
    restored = ScenarioConfig.from_json(cfg.to_json())
    assert restored.guard_action_components == cfg.guard_action_components
    assert restored.bandit_action_components == cfg.bandit_action_components
