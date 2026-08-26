"""End-to-end attitude+cone integration: belief grows when target is out
of cone; belief contracts when guard's cone faces the target.

Validates the full loop: attitude dynamics -> cone gate -> belief update.

Both tests use a 1v1 RT scenario with attitude state + ATTITUDE_CONTROL + a
conical guard observation. The guard's initial quaternion and bandit position
are pinned post-reset so the geometry is deterministic.

Quaternion convention: (w, x, y, z). Identity = (1, 0, 0, 0) gives
rotation matrix R = I, so the body-fixed boresight (+x) maps to +x world.
180-deg rotation about z = (0, 0, 0, 1) gives R[0, 0] = -1, so the
boresight maps to -x world.

NOTE: belief_rollout() requires IMPULSIVE_MANEUVER (reads .dv for EKF
prediction). Because this scenario uses ATTITUDE_CONTROL only, the belief
loop is hand-rolled with env.step + guard_observation_fn + ekf_updater.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from orbitalgym.belief.ekf import EKFBeliefUpdater, EKFFromTruthInitializer
from orbitalgym.config import ScenarioConfig
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.dynamics.hcw import hcw_rt_step
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.reference_orbit import mean_motion as ref_mean_motion
from orbitalgym.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from tests.helpers.minimal_scenario import minimal_scenario_kwargs

# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

_N_STEPS = 20
_DT = 10.0
_HALF_ANGLE_RAD = float(jnp.deg2rad(30.0))  # 30-degree half-angle cone
_BORESIGHT = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)  # +x body


def _attitude_rt_kwargs():
    """1v1 RT scenario with rigid-body attitude state and ATTITUDE_CONTROL."""
    return minimal_scenario_kwargs(
        truth_dynamics=DynamicsKey.HCW_RT,
        action_frame=Frame.RT,
        guard_components=(
            StateComponentKey.RT,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        bandit_components=(
            StateComponentKey.RT,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        guard_action_components=(ActionComponentKey.ATTITUDE_CONTROL,),
        bandit_action_components=(),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=AttitudeParams(
            inertia_diag=jnp.ones(3) * 1.0,
            omega_max=jnp.ones(3) * 0.5,
        ),
        bandit_attitude_params=AttitudeParams(
            inertia_diag=jnp.ones(3) * 1.0,
            omega_max=jnp.ones(3) * 0.5,
        ),
        dt=_DT,
        max_horizon_s=float(_N_STEPS * _DT + _DT),  # enough steps
    )


def _make_env_with_cone():
    """Build env + conical obs fn. Returns (env, obs_fn)."""
    cfg = ScenarioConfig(**_attitude_rt_kwargs())
    obs_fn = ConicalObservation(
        layout=cfg.layout,
        sensor_boresights_body=_BORESIGHT,
        half_angle_rad=_HALF_ANGLE_RAD,
        sigma=0.1,
    )
    cfg_with_cone = ScenarioConfig(
        **{
            **_attitude_rt_kwargs(),
            "guard_observation_fn": obs_fn,
        }
    )
    env = OrbitalGymEnv(cfg_with_cone)
    return env, obs_fn


def _build_ekf(env):
    """EKF updater + initializer for the 4-D RT state."""
    n_motion = float(ref_mean_motion(env.config.reference_orbit))
    hcw_params = SimpleNamespace(mean_motion=n_motion)

    def per_vehicle_dyn(x, u, dt):
        # x: (4,), u: (2,) -> (4,)
        return hcw_rt_step(x[None, :], u[None, :], hcw_params, dt)[0]

    d = env.layout.dynamics_state_dim  # 4 for RT
    updater = EKFBeliefUpdater(
        dynamics_fn=per_vehicle_dyn,
        process_noise=jnp.eye(d, dtype=jnp.float32) * 0.1,
        dt=_DT,
    )
    initializer = EKFFromTruthInitializer(
        layout=env.layout,
        variance_diag=jnp.ones(d, dtype=jnp.float32) * 50.0,
    )
    return updater, initializer


def _pin_state(state, guard_quat, bandit_rt):
    """Pin guard quaternion and bandit RT position in the initial state."""
    guards = state.guards.replace(
        quat=jnp.array([guard_quat], dtype=jnp.float32),
        omega=jnp.zeros((1, 3), dtype=jnp.float32),
    )
    # Overwrite bandit RT (radial=x, tangential=y): place at known position.
    new_rt = state.bandits.rt.at[0, :].set(jnp.asarray(bandit_rt, dtype=jnp.float32))
    bandits = state.bandits.replace(rt=new_rt)
    return state.replace(guards=guards, bandits=bandits)


def _rollout_belief(env, obs_fn, updater, initializer, state, n_steps):
    """Hand-rolled belief loop: env.step + manual EKF update.

    Returns list of Frobenius norms of the guard's belief covariance
    block for the bandit (index 1 in the n_total axis), one per step.
    The bandit-block has shape (d, d).
    """
    n_guards = env.config.n_guards
    key = jax.random.PRNGKey(42)

    # Initialise belief from the pinned truth state.
    belief = initializer(state, Side.GUARD, key)

    zero_guard_cmd = env.guard_command_cls.zeros(n_guards)
    zero_bandit_cmd = env.bandit_command_cls.zeros(env.config.n_bandits)
    identity_actions = Actions(sides=BySide(guard=zero_guard_cmd, bandit=zero_bandit_cmd))

    # Zero translational action for EKF prediction (no IMPULSIVE_MANEUVER).
    zero_dv = jnp.zeros((n_guards, 2), dtype=jnp.float32)

    cov_norms = []
    cur_state = state

    for _i in range(n_steps):
        k_step, k_obs, k_upd, key = jax.random.split(key, 4)

        step_out = env.step(k_step, cur_state, identity_actions)
        cur_state = step_out.state

        # Observe with the cone sensor on the new state.
        obs_channels = obs_fn(
            cur_state, identity_actions, Side.GUARD, env.config, k_obs, cur_state.t
        )

        # Update belief.
        belief = updater(belief, obs_channels, zero_dv, Side.GUARD, k_upd)

        # Record Frobenius norm of guard's bandit-block covariance.
        # belief.cov: (n_guards, n_total, d, d); bandit is index n_guards (=1).
        bandit_cov = belief.cov[0, n_guards]  # (d, d)
        cov_norms.append(float(jnp.linalg.norm(bandit_cov, ord="fro")))

    return cov_norms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_belief_grows_when_target_is_out_of_cone():
    """Guard boresight points -x; bandit is at +x => never in cone.

    Belief covariance should grow monotonically (only predict steps, no
    corrections) or at least be larger at the end than at the start.
    """
    env, obs_fn = _make_env_with_cone()
    updater, initializer = _build_ekf(env)

    # Reset and pin geometry.
    state, _ = env.reset(jax.random.PRNGKey(0))

    # Guard quat: 180-deg about z => boresight +x body -> -x world.
    quat_180z = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    # Bandit at +x (radial direction, 500 m), well outside the -x boresight cone.
    bandit_pos = [500.0, 0.0, 0.0, 0.0]  # (r, t, rdot, tdot) = RT state

    state = _pin_state(state, guard_quat=quat_180z, bandit_rt=bandit_pos)

    cov_norms = _rollout_belief(env, obs_fn, updater, initializer, state, n_steps=_N_STEPS)

    # Sanity: first norm should be finite and positive.
    assert cov_norms[0] > 0.0, "Initial covariance norm should be positive"

    # With no corrections, covariance grows via process noise at every step.
    assert cov_norms[-1] > cov_norms[0], (
        f"Belief covariance should grow when target is out of cone: "
        f"cov_norm[0]={cov_norms[0]:.4f}, cov_norm[-1]={cov_norms[-1]:.4f}"
    )


def test_belief_contracts_when_target_is_in_cone():
    """Guard boresight is +x (identity quat); bandit is at +x => in cone at t=0.

    Belief covariance should contract at least once early in the rollout
    (correction steps dominate process noise for the first few steps).
    """
    env, obs_fn = _make_env_with_cone()
    updater, initializer = _build_ekf(env)

    state, _ = env.reset(jax.random.PRNGKey(1))

    # Guard quat: identity => boresight +x body = +x world.
    quat_identity = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    # Bandit at +x (500 m) — same side as boresight.
    bandit_pos = [500.0, 0.0, 0.0, 0.0]

    state = _pin_state(state, guard_quat=quat_identity, bandit_rt=bandit_pos)

    # Initialise belief with large variance so there is room to contract.
    cov_norms = _rollout_belief(env, obs_fn, updater, initializer, state, n_steps=_N_STEPS)

    assert cov_norms[0] > 0.0, "Initial covariance norm should be positive"

    # At least one of the first 5 steps should shrink the covariance.
    initial_norm = cov_norms[0]
    min_early = min(cov_norms[:5])
    assert min_early < initial_norm, (
        f"Belief covariance should contract when target is in cone: "
        f"initial={initial_norm:.4f}, min_early={min_early:.4f}. "
        f"First 5 norms: {[f'{v:.4f}' for v in cov_norms[:5]]}"
    )
