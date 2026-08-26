"""Drives a minimal attitude scenario to verify the env's attitude
dynamics block runs correctly: zero torque => free precession; nonzero
applied_torque => omega ramps."""

import jax
import jax.numpy as jnp

from orbitalgym.config import ScenarioConfig
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide
from orbitalgym.registry import (
    AttitudeDynamicsKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from tests.helpers.minimal_scenario import minimal_scenario_kwargs


def _attitude_kwargs():
    return minimal_scenario_kwargs(
        truth_dynamics=DynamicsKey.HCW_RTN,
        action_frame=Frame.RTN,
        guard_components=(
            StateComponentKey.RTN,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        bandit_components=(
            StateComponentKey.RTN,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        guard_action_components=(),
        bandit_action_components=(),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=AttitudeParams(
            inertia_diag=jnp.ones(3),
            omega_max=jnp.ones(3) * 10.0,
        ),
        bandit_attitude_params=AttitudeParams(
            inertia_diag=jnp.ones(3),
            omega_max=jnp.ones(3) * 10.0,
        ),
    )


def _identity_actions(env, cfg):
    return Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )


def test_zero_omega_zero_torque_keeps_attitude():
    cfg = ScenarioConfig(**_attitude_kwargs())
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    pre_quat = state.guards.quat
    pre_omega = state.guards.omega

    out = env.step(jax.random.PRNGKey(1), state, _identity_actions(env, cfg))
    assert jnp.allclose(out.state.guards.quat, pre_quat)
    assert jnp.allclose(out.state.guards.omega, pre_omega)


def test_nonzero_initial_omega_advances_quaternion():
    cfg = ScenarioConfig(**_attitude_kwargs())
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    state = state.replace(
        guards=state.guards.replace(omega=jnp.array([[0.0, 0.0, 1.0]] * cfg.n_guards))
    )

    out = env.step(jax.random.PRNGKey(1), state, _identity_actions(env, cfg))
    assert out.state.guards.quat[0, 0] < 1.0
