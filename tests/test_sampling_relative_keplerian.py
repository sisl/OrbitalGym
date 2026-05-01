"""Tests for RelativeKeplerian per-side sampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import StateComponentKey
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeKeplerian


def _config(n_guards=2, n_bandits=1):
    # Minimal config — only fields the sampler reads matter.
    from orbital_game.sampling.side import RelativeEllipse
    from orbital_game.sampling.spec import ICSpec
    return ScenarioConfig(
        n_guards=n_guards,
        n_bandits=n_bandits,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7546.05, 1.0]),
        ),
        guard_components=(StateComponentKey.RTN,),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=VehicleParamsSpec(100.0, 220.0, 5.0),
        bandit_params=VehicleParamsSpec(50.0, 200.0, 2.0),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(radial_ellipse_m=0.0, phase_rad=0.0),
            bandit_sampler=RelativeEllipse(radial_ellipse_m=0.0, phase_rad=0.0),
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )


def test_zero_sigma_zero_mean_yields_origin_state():
    sampler = RelativeKeplerian()  # all defaults zero
    out = sampler(
        _config(),
        jax.random.PRNGKey(0),
        n_vehicles=2,
        components=(StateComponentKey.RTN,),
        class_name="GuardState",
    )
    # All vehicles co-orbital with reference orbit -> RTN state ~ 0.
    assert jnp.allclose(out.rtn, jnp.zeros((2, 6)), atol=1.0)


def test_sigma_sma_produces_radial_offset_distribution():
    sampler = RelativeKeplerian(sigma_delta_sma_m=100.0)
    out = sampler(
        _config(n_guards=64),
        jax.random.PRNGKey(0),
        n_vehicles=64,
        components=(StateComponentKey.RTN,),
        class_name="GuardState",
    )
    radial = out.rtn[:, 0]
    assert jnp.std(radial) > 1.0
    assert jnp.std(radial) < 1e6


def test_per_vehicle_sigma_sma_array():
    sigmas = jnp.array([0.0, 200.0])
    sampler = RelativeKeplerian(sigma_delta_sma_m=sigmas)
    out = sampler(
        _config(n_guards=2),
        jax.random.PRNGKey(7),
        n_vehicles=2,
        components=(StateComponentKey.RTN,),
        class_name="GuardState",
    )
    # Vehicle 0 has zero sigma_delta_sma, so its radial offset should be ~0,
    # not approach the second vehicle's sampled offset (which scales with
    # sigma=200 m).
    assert jnp.abs(out.rtn[0, 0]) < 1.0
    assert out.rtn.shape == (2, 6)


def test_mass_sampler_populates_propellant_when_mass_in_components():
    sampler = RelativeKeplerian(mass_sampler=ConstantMass(propellant_mass_kg=7.5))
    out = sampler(
        _config(n_guards=3),
        jax.random.PRNGKey(0),
        n_vehicles=3,
        components=(StateComponentKey.RTN, StateComponentKey.MASS),
        class_name="GuardState",
    )
    assert jnp.allclose(out.propellant_mass, 7.5)


def test_mass_required_when_mass_in_components_else_raises():
    sampler = RelativeKeplerian()
    with pytest.raises(ValueError, match="mass_sampler"):
        sampler(
            _config(n_guards=1),
            jax.random.PRNGKey(0),
            n_vehicles=1,
            components=(StateComponentKey.RTN, StateComponentKey.MASS),
            class_name="GuardState",
        )


def test_precision_floor_is_sub_meter():
    """Regression: with astrojax float64, KOE round-trip residual on the test
    reference orbit is sub-mm, not the float32 default of ~7m. If this test ever
    shows meter-scale residuals, astrojax dtype config has regressed."""
    sampler = RelativeKeplerian()  # zero deltas -> co-orbital with reference orbit
    out = sampler(
        _config(),
        jax.random.PRNGKey(0),
        n_vehicles=1,
        components=(StateComponentKey.RTN,),
        class_name="GuardState",
    )
    rtn = out.rtn[0]
    # Co-orbital vehicle: relative state should be ~zero. Float32 gives ~7m
    # residual; float64 gives sub-mm. Threshold of 0.01m catches a regression
    # while leaving headroom for legitimate numerical noise.
    assert float(jnp.linalg.norm(rtn[:3])) < 0.01, f"radial+along+cross: {rtn[:3]}"
    assert float(jnp.linalg.norm(rtn[3:])) < 1e-5, f"velocity: {rtn[3:]}"
