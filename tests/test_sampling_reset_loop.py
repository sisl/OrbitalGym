"""Tests for env.reset's rejection-sampling loop and ic_valid propagation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.env.environment import OrbitalGameEnv
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import StateComponentKey
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec


def _bounded_ellipse_config(validators=()):
    return ScenarioConfig(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7546.05, 0.0]),
        ),
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=VehicleParamsSpec(100.0, 220.0, 5.0),
        bandit_params=VehicleParamsSpec(50.0, 200.0, 2.0),
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
            validators=validators,
            max_attempts=50,
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )


def test_reset_returns_ic_valid_true_when_no_validators():
    env = OrbitalGameEnv(_bounded_ellipse_config())
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert bool(state.ic_valid)


def test_reset_returns_ic_valid_false_when_always_failing_validator():
    @dataclass(frozen=True)
    class _AlwaysFalse:
        def __call__(self, config, guards, bandits):
            return jnp.asarray(False)
    env = OrbitalGameEnv(_bounded_ellipse_config(validators=(_AlwaysFalse(),)))
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert not bool(state.ic_valid)


def test_reset_returns_ic_valid_true_when_passing_validator():
    @dataclass(frozen=True)
    class _AlwaysTrue:
        def __call__(self, config, guards, bandits):
            return jnp.asarray(True)
    env = OrbitalGameEnv(_bounded_ellipse_config(validators=(_AlwaysTrue(),)))
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert bool(state.ic_valid)


def test_reset_under_vmap_produces_per_lane_ic_valid():
    @dataclass(frozen=True)
    class _AlwaysFalse:
        def __call__(self, config, guards, bandits):
            return jnp.asarray(False)
    env = OrbitalGameEnv(_bounded_ellipse_config(validators=(_AlwaysFalse(),)))

    def reset_fn(k):
        s, _ = env.reset(k)
        return s.ic_valid

    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    valids = jax.vmap(reset_fn)(keys)
    assert valids.shape == (8,)
    assert not bool(jnp.any(valids))  # all lanes hit max_attempts
