"""A PointAt command slews the guard boresight through env.step."""

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.config import VehicleParamsSpec
from orbitalgym.dynamics.quaternion import quat_to_rotation_matrix
from orbitalgym.env.types import Actions, BySide
from orbitalgym.registry import ActionComponentKey, AttitudeDynamicsKey, StateComponentKey


def test_env_step_slews_toward_target():
    cfg = make_lady_bandit_guard(
        guard_components=(
            StateComponentKey.RTN,
            StateComponentKey.MASS,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        attitude_dynamics_key=AttitudeDynamicsKey.KINEMATIC,
        guard_action_components=(
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.POINT_AT,
        ),
        guard_params=VehicleParamsSpec(100.0, 220.0, 5.0, slew_rate_rad_s=jnp.deg2rad(1.0)),
    )
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    guard_cmd = env.guard_command_cls.zeros(1).replace(target_dir=jnp.array([[0.0, 1.0, 0.0]]))
    actions = Actions(sides=BySide(guard=guard_cmd, bandit=env.bandit_command_cls.zeros(1)))
    out = env.step(jax.random.PRNGKey(1), state, actions)
    b = quat_to_rotation_matrix(out.state.guards.quat[0]) @ jnp.array([1.0, 0.0, 0.0])
    assert jnp.allclose(jnp.arccos(jnp.clip(b[0], -1, 1)), jnp.deg2rad(10.0), atol=1e-5)
    assert jnp.allclose(out.state.guards.omega, 0.0)
