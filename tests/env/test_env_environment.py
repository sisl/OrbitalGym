"""Tests for env/core.py — OrbitalGameEnv reset/step determinism + shape contract."""

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide
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


def _make_actions(cfg: ScenarioConfig, guard_action: jnp.ndarray) -> Actions:
    return Actions(
        sides=BySide(
            guard=guard_action,
            bandit=jnp.zeros((cfg.n_bandits, 3)),
        )
    )


def test_env_reset_is_deterministic_under_same_key():
    from orbital_game.observations.types import flatten_observations

    env = OrbitalGameEnv(_make_cfg())
    key = jax.random.PRNGKey(42)
    s_a, outputs_a = env.reset(key)
    s_b, outputs_b = env.reset(key)
    assert jnp.allclose(s_a.guards.rtn, s_b.guards.rtn)
    assert jnp.allclose(
        flatten_observations(outputs_a.guard.obs),
        flatten_observations(outputs_b.guard.obs),
    )


def test_env_step_is_deterministic_under_same_key():
    cfg = _make_cfg()
    env = OrbitalGameEnv(cfg)
    key = jax.random.PRNGKey(42)
    s0, _ = env.reset(key)
    actions = _make_actions(cfg, jnp.zeros((1, 3)))
    step_key = jax.random.PRNGKey(99)
    out_a = env.step(step_key, s0, actions)
    out_b = env.step(step_key, s0, actions)
    assert jnp.allclose(out_a.state.guards.rtn, out_b.state.guards.rtn)
    assert jnp.isclose(out_a.outputs.guard.reward, out_b.outputs.guard.reward)


def test_env_step_increments_step_counter_and_time():
    cfg = _make_cfg()
    env = OrbitalGameEnv(cfg)
    key = jax.random.PRNGKey(0)
    s0, _ = env.reset(key)
    assert int(s0.step) == 0
    assert float(s0.t) == 0.0
    actions = _make_actions(cfg, jnp.zeros((1, 3)))
    step_out = env.step(jax.random.PRNGKey(1), s0, actions)
    assert int(step_out.state.step) == 1
    assert float(step_out.state.t) == 10.0  # dt


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
    s0, outputs0 = env.reset(jax.random.PRNGKey(0))
    assert s0.guards.rt.shape == (1, 4)
    assert s0.bandits.rt.shape == (1, 4)
    # Guard action is 2D in RT scenarios.
    actions = Actions(
        sides=BySide(
            guard=jnp.zeros((1, 2)),
            bandit=jnp.zeros((1, 2)),
        )
    )
    step_out = env.step(jax.random.PRNGKey(1), s0, actions)
    assert step_out.state.guards.rt.shape == (1, 4)
    assert jnp.isfinite(step_out.outputs.guard.reward)
    assert bool(step_out.episode_done) in (True, False)  # termination is a bool scalar
