"""Tests for ImpulsiveActuator: rocket-equation propellant depletion + track_mass gate."""

import jax.numpy as jnp

from orbital_game.actuators.impulsive import G0, ImpulsiveActuator
from orbital_game.config import VehicleParamsSpec


class _State:
    def __init__(self, propellant_mass):
        self.propellant_mass = propellant_mass


def test_impulsive_tracked_mass_delta_matches_rocket_equation():
    params = VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0)
    state = _State(propellant_mass=jnp.array([10.0]))
    command = jnp.array([[3.0, 4.0, 0.0]])  # |dv| = 5 m/s
    act = ImpulsiveActuator(track_mass=True)
    applied, dp = act.apply(command, state, params, dt=1.0)

    dv_mag = 5.0
    wet = 100.0 + 10.0
    expected_dp = wet * (1.0 - jnp.exp(-dv_mag / (220.0 * G0)))
    assert jnp.allclose(dp[0], expected_dp, atol=1e-6)
    assert jnp.allclose(applied.dv, command)


def test_impulsive_untracked_mass_returns_zeros():
    params = VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0)
    state = _State(propellant_mass=jnp.zeros((3,)))
    command = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    act = ImpulsiveActuator(track_mass=False)
    applied, dp = act.apply(command, state, params, dt=1.0)
    assert jnp.allclose(dp, jnp.zeros((3,)))
    assert jnp.allclose(applied.dv, command)
