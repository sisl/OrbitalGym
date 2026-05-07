"""Integration tests: attitude_sampler field plumbed through RelativeEllipse."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig
from orbital_game.dynamics.attitude import AttitudeParams
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.registry import (
    AttitudeDynamicsKey,
    StateComponentKey,
)
from orbital_game.sampling.attitude import FixedAttitude, UniformBodyRates
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec
from tests.helpers.minimal_scenario import minimal_scenario_kwargs


def _attitude_components():
    return (StateComponentKey.RTN, StateComponentKey.ATTITUDE, StateComponentKey.BODY_RATES)


def _attitude_params():
    return AttitudeParams(
        inertia_diag=jnp.ones(3, dtype=jnp.float32),
        omega_max=jnp.ones(3, dtype=jnp.float32),
    )


def test_fixed_attitude_threads_to_state():
    omega_z = float(jnp.deg2rad(0.1))
    sampler_g = RelativeEllipse(
        radial_ellipse_m=10.0,
        attitude_sampler=FixedAttitude(
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            omega_rad_s=(0.0, 0.0, omega_z),
        ),
    )
    sampler_b = RelativeEllipse(radial_ellipse_m=10.0)  # default = IdentityAttitude
    cfg_kwargs = minimal_scenario_kwargs(
        guard_components=_attitude_components(),
        bandit_components=_attitude_components(),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=_attitude_params(),
        bandit_attitude_params=_attitude_params(),
        ic_sampler=ICSpec(
            guard_sampler=sampler_g,
            bandit_sampler=sampler_b,
            validators=(),
            max_attempts=1,
        ),
    )
    cfg = ScenarioConfig(**cfg_kwargs)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    # Guard pinned to the requested rate.
    assert jnp.allclose(state.guards.omega[:, 2], omega_z, atol=1e-6), (
        f"Guard omega_z mismatch: {state.guards.omega[:, 2]} vs {omega_z}"
    )
    # Bandit gets default IdentityAttitude → zero omega.
    assert jnp.allclose(state.bandits.omega, 0.0), (
        f"Bandit omega should be zero, got {state.bandits.omega}"
    )


def test_uniform_body_rates_within_bound():
    sampler = RelativeEllipse(
        radial_ellipse_m=10.0,
        attitude_sampler=UniformBodyRates(omega_max_rad_s=(0.5, 0.5, 0.5)),
    )
    cfg_kwargs = minimal_scenario_kwargs(
        guard_components=_attitude_components(),
        bandit_components=_attitude_components(),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=_attitude_params(),
        bandit_attitude_params=_attitude_params(),
        ic_sampler=ICSpec(
            guard_sampler=sampler,
            bandit_sampler=sampler,
            validators=(),
            max_attempts=1,
        ),
    )
    cfg = ScenarioConfig(**cfg_kwargs)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert bool(jnp.all(jnp.abs(state.guards.omega) <= 0.5)), (
        f"Guard omega out of bounds: {state.guards.omega}"
    )


def test_no_attitude_sampler_defaults_to_identity():
    """When attitude_sampler=None (default), IC has identity quat + zero omega."""
    sampler = RelativeEllipse(radial_ellipse_m=10.0)  # no attitude_sampler
    cfg_kwargs = minimal_scenario_kwargs(
        guard_components=_attitude_components(),
        bandit_components=_attitude_components(),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=_attitude_params(),
        bandit_attitude_params=_attitude_params(),
        ic_sampler=ICSpec(
            guard_sampler=sampler,
            bandit_sampler=sampler,
            validators=(),
            max_attempts=1,
        ),
    )
    cfg = ScenarioConfig(**cfg_kwargs)
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    # Identity quaternion: w=1, xyz=0.
    expected_quat = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    assert jnp.allclose(state.guards.quat, expected_quat, atol=1e-6), (
        f"Expected identity quat, got {state.guards.quat}"
    )
    assert jnp.allclose(state.guards.omega, 0.0), f"Expected zero omega, got {state.guards.omega}"
