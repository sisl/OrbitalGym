"""Tests for ScenarioConfig.reference_orbit_dynamics resolution (Task 5.1)."""

from __future__ import annotations

import pytest


def _minimal_cfg(**overrides):
    """Minimal HCW_RTN scenario for testing reference-dynamics resolution.

    Constructs a ScenarioConfig with the smallest valid state. Uses RTN-frame
    components by default; mixed-frame tests pass overrides for guard/bandit
    components and dynamics.
    """
    import jax.numpy as jnp

    from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
    from orbitalgym.reference_orbit import ReferenceOrbitState
    from orbitalgym.registry import StateComponentKey
    from orbitalgym.sampling.side import RelativeEllipse
    from orbitalgym.sampling.spec import ICSpec

    ref = ReferenceOrbitState(
        position_eci=jnp.array([6878137.0, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7612.7, 0.0]),
    )
    params = VehicleParamsSpec(dry_mass_kg=10.0, isp_s=200.0, max_thrust_n=1.0)
    sampler = RelativeEllipse(radial_ellipse_m=0.0, phase_rad=0.0)
    base = dict(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=58849.0,
        reference_orbit=ref,
        guard_components=(StateComponentKey.RTN,),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=params,
        bandit_params=params,
        ic_sampler=ICSpec(
            guard_sampler=sampler,
            bandit_sampler=sampler,
            validators=(),
            max_attempts=10,
        ),
        dt=10.0,
        max_horizon_s=100.0,
        seed=0,
    )
    base.update(overrides)
    return ScenarioConfig(**base)


def test_reference_orbit_dynamics_default_for_relative_truth():
    """Truth=HCW_RTN (relative) -> reference defaults to a KeplerianEciDynamics
    instance that carries the scenario's epoch (so different scenarios get
    different reference-epoch instances rather than sharing module state)."""
    from orbitalgym.dynamics.keplerian import KeplerianEciDynamics

    cfg = _minimal_cfg()
    assert isinstance(cfg.reference_orbit_dynamics_resolved, KeplerianEciDynamics)
    assert cfg.reference_orbit_dynamics_resolved.epoch_mjd_utc == cfg.epoch_mjd_utc


def test_reference_orbit_dynamics_default_for_absolute_truth():
    """Truth=KEPLERIAN_ECI (absolute) -> reference defaults to KEPLERIAN_ECI (= truth)."""
    from orbitalgym.registry import DynamicsKey, StateComponentKey

    cfg = _minimal_cfg(
        truth_dynamics=DynamicsKey.KEPLERIAN_ECI,
        policy_dynamics=DynamicsKey.KEPLERIAN_ECI,
        guard_components=(StateComponentKey.ECI,),
        bandit_components=(StateComponentKey.ECI,),
    )
    assert cfg.reference_orbit_dynamics_resolved is DynamicsKey.KEPLERIAN_ECI


def test_reference_orbit_dynamics_explicit_relative_rejected():
    """Explicitly setting reference to a relative-kind dynamics is an error."""
    from orbitalgym.registry import DynamicsKey

    with pytest.raises(ValueError, match="reference_orbit_dynamics.*ABSOLUTE"):
        _minimal_cfg(reference_orbit_dynamics=DynamicsKey.HCW_RTN)
