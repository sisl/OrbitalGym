"""Minimal ScenarioConfig kwargs helper for tests.

Returns every required kwarg for ``ScenarioConfig(**kwargs)`` with sensible
defaults. Call ``minimal_scenario_kwargs(**overrides)`` and pass the result
directly to ``ScenarioConfig``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from orbitalgym.config import VehicleParamsSpec
from orbitalgym.reference_orbit import ReferenceOrbitState
from orbitalgym.registry import (
    ActionComponentKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec


def minimal_scenario_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a minimal-valid dict of kwargs for ``ScenarioConfig``.

    Defaults: HCW_RTN truth dynamics, RTN frame, 1 guard + 1 bandit,
    impulsive maneuver action components, 10 s step, 100 s horizon.
    Pass overrides to change any field.
    """
    ref = ReferenceOrbitState(
        position_eci=jnp.array([6878137.0, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7612.7, 0.0]),
    )
    params = VehicleParamsSpec(dry_mass_kg=10.0, isp_s=200.0, max_thrust_n=1.0)
    sampler = RelativeEllipse(radial_ellipse_m=0.0, phase_rad=0.0)

    base: dict[str, Any] = dict(
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
        truth_dynamics=DynamicsKey.HCW_RTN,
        action_frame=Frame.RTN,
        guard_action_components=(ActionComponentKey.IMPULSIVE_MANEUVER,),
        bandit_action_components=(ActionComponentKey.IMPULSIVE_MANEUVER,),
    )
    base.update(overrides)
    return base
