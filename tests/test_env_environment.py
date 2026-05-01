"""Tests for env/environment.py — OrbitalGameEnv reset/step determinism + shape contract."""

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.env.environment import OrbitalGameEnv
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import DynamicsKey, StateComponentKey
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec


def _make_cfg() -> ScenarioConfig:
    return ScenarioConfig(
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
                phase_rad=0.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=jnp.pi,
            ),
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
    )


def test_env_reset_is_deterministic_under_same_key():
    env = OrbitalGameEnv(_make_cfg())
    key = jax.random.PRNGKey(42)
    s_a, obs_a = env.reset(key)
    s_b, obs_b = env.reset(key)
    assert jnp.allclose(s_a.guards.rtn, s_b.guards.rtn)
    assert jnp.allclose(obs_a, obs_b)


def test_env_step_is_deterministic_under_same_key():
    env = OrbitalGameEnv(_make_cfg())
    key = jax.random.PRNGKey(42)
    s0, _ = env.reset(key)
    action = jnp.zeros((1, 3))
    step_key = jax.random.PRNGKey(99)
    s1_a, obs_a, r_a, d_a, _ = env.step(step_key, s0, action)
    s1_b, obs_b, r_b, d_b, _ = env.step(step_key, s0, action)
    assert jnp.allclose(s1_a.guards.rtn, s1_b.guards.rtn)
    assert jnp.isclose(r_a, r_b)


def test_env_step_increments_step_counter_and_time():
    env = OrbitalGameEnv(_make_cfg())
    key = jax.random.PRNGKey(0)
    s0, _ = env.reset(key)
    assert int(s0.step) == 0
    assert float(s0.t) == 0.0
    s1, _, _, _, _ = env.step(jax.random.PRNGKey(1), s0, jnp.zeros((1, 3)))
    assert int(s1.step) == 1
    assert float(s1.t) == 10.0  # dt


def test_env_runs_end_to_end_with_rt_2d_dynamics():
    """HCW_RT (2D) scenario: reward + termination use 2D norm over (r, t), no error."""
    cfg = ScenarioConfig(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        guard_components=(StateComponentKey.RT,),
        bandit_components=(StateComponentKey.RT,),
        guard_params=VehicleParamsSpec(100.0, 220.0, 5.0),
        bandit_params=VehicleParamsSpec(50.0, 200.0, 2.0),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=0.0,
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                phase_rad=jnp.pi,
            ),
        ),
        dt=10.0,
        max_horizon_s=2000.0,
        seed=0,
        truth_dynamics=DynamicsKey.HCW_RT,
        planning_dynamics=DynamicsKey.HCW_RT,
    )
    env = OrbitalGameEnv(cfg)
    s0, obs0 = env.reset(jax.random.PRNGKey(0))
    assert s0.guards.rt.shape == (1, 4)
    assert s0.bandits.rt.shape == (1, 4)
    # Guard action is 2D in RT scenarios.
    s1, obs1, r1, d1, _ = env.step(jax.random.PRNGKey(1), s0, jnp.zeros((1, 2)))
    assert s1.guards.rt.shape == (1, 4)
    assert jnp.isfinite(r1)
    assert bool(d1) in (True, False)  # termination is a bool scalar
