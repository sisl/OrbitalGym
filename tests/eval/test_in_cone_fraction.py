"""in_cone_fraction_guard metric and traj.visible logging in belief_rollout."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from orbitalgym.belief.kf import KFBeliefUpdater, KFFromTruthInitializer
from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide
from orbitalgym.eval.metrics import lbg_episode_metrics
from orbitalgym.games.lady_bandit_guard import LadyBanditGuard
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.policies.pointing import PointingPolicy, PointingTarget
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from orbitalgym.rollout import belief_rollout
from tests.helpers.minimal_scenario import minimal_scenario_kwargs

_N_STEPS = 30
_DT = 10.0
_HALF_ANGLE_RAD = float(jnp.deg2rad(5.0))


def test_full_observation_gives_in_cone_fraction_one(make_minimal_lbg_env_with_kf):
    env, belief_init, belief_upd, init_ps_fns, policies = make_minimal_lbg_env_with_kf()
    traj, _ = belief_rollout(
        env=env,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
        key=jax.random.PRNGKey(0),
        n_steps=5,
    )
    assert traj.visible.guard.shape == (5, env.config.n_guards)
    assert bool(jnp.all(traj.visible.guard))
    m = lbg_episode_metrics(traj, env.config)
    assert float(m.in_cone_fraction_guard) == 1.0


def _cone_pointed_away_kwargs():
    """Guard: RT + ATTITUDE/BODY_RATES + POINT_AT, held facing +x while the
    bandit sits at -x — always outside a 5-degree cone."""
    return minimal_scenario_kwargs(
        truth_dynamics=DynamicsKey.HCW_RT,
        action_frame=Frame.RT,
        guard_components=(
            StateComponentKey.RT,
            StateComponentKey.ATTITUDE,
            StateComponentKey.BODY_RATES,
        ),
        bandit_components=(StateComponentKey.RT,),
        guard_action_components=(
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.POINT_AT,
        ),
        bandit_action_components=(ActionComponentKey.IMPULSIVE_MANEUVER,),
        attitude_dynamics_key=AttitudeDynamicsKey.RIGID_BODY,
        guard_attitude_params=AttitudeParams(inertia_diag=jnp.ones(3), omega_max=jnp.ones(3) * 0.5),
        pointing_boresight_body=(1.0, 0.0, 0.0),
        guard_params=VehicleParamsSpec(
            dry_mass_kg=10.0, isp_s=200.0, max_thrust_n=1.0, slew_rate_rad_s=0.1
        ),
        bandit_params=VehicleParamsSpec(dry_mass_kg=10.0, isp_s=200.0, max_thrust_n=1.0),
        dt=_DT,
        max_horizon_s=_N_STEPS * _DT + _DT,
        game=LadyBanditGuard(breach_radius_m=5.0, catch_radius_m=50.0),
    )


def _pin_geometry(state):
    """Guard boresight (+x body, identity quat) faces +x world; bandit at -x."""
    quat_dtype = state.guards.quat.dtype
    identity_quat = jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=quat_dtype)
    guards = state.guards.replace(
        quat=identity_quat, omega=jnp.zeros((1, 3), dtype=state.guards.omega.dtype)
    )
    new_rt = state.bandits.rt.at[0, :].set(
        jnp.array([-500.0, 0.0, 0.0, 0.0], dtype=state.bandits.rt.dtype)
    )
    bandits = state.bandits.replace(rt=new_rt)
    return state.replace(guards=guards, bandits=bandits)


def test_conical_sensor_pointed_away_gives_in_cone_fraction_zero():
    cfg = ScenarioConfig(
        **{
            **_cone_pointed_away_kwargs(),
        }
    )
    obs_fn = ConicalObservation(
        layout=cfg.layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]]),
        half_angle_rad=_HALF_ANGLE_RAD,
        sigma_floor=0.1,
    )
    cfg = dataclasses.replace(cfg, guard_observation_fn=obs_fn)
    env = OrbitalGymEnv(cfg)

    d = env.layout.dynamics_state_dim
    belief_init = BySide(
        guard=KFFromTruthInitializer(layout=env.layout, variance_diag=jnp.ones(d)),
        bandit=KFFromTruthInitializer(layout=env.layout, variance_diag=jnp.ones(d)),
    )
    kf_upd = KFBeliefUpdater(
        stm=jnp.eye(d),
        control_matrix=jnp.zeros((d, 2)),
        process_noise=jnp.eye(d) * 0.01,
    )
    belief_upd = BySide(guard=kf_upd, bandit=kf_upd)

    guard_zero = dataclasses.replace(
        ZeroControl(), command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards
    )
    guard_policy = PointingPolicy(inner=guard_zero, target=PointingTarget.HOLD)
    bandit_policy = dataclasses.replace(
        ZeroControl(), command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits
    )
    policies = BySide(guard=guard_policy, bandit=bandit_policy)
    init_ps_fns = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)

    state, _ = env.reset(jax.random.PRNGKey(0))
    state = _pin_geometry(state)

    traj, _ = belief_rollout(
        env=env,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
        key=jax.random.PRNGKey(1),
        n_steps=_N_STEPS,
        initial_state=state,
    )
    assert traj.visible.guard.shape == (_N_STEPS, cfg.n_guards)
    assert not bool(jnp.any(traj.visible.guard))
    m = lbg_episode_metrics(traj, cfg)
    assert float(m.in_cone_fraction_guard) == 0.0
