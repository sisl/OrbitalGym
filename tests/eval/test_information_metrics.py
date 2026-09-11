"""Guard information metrics: belief error, detection time, and belief age."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.belief.kf import (
    KFBeliefUpdater,
    KFFromTruthInitializer,
    KFUniformDefaultInitializer,
)
from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
from orbitalgym.dynamics.attitude import AttitudeParams
from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide
from orbitalgym.eval.bank import sample_bank
from orbitalgym.eval.evaluate import evaluate_bank, metrics_to_records
from orbitalgym.eval.metrics import guard_information, lbg_episode_metrics
from orbitalgym.games.lady_bandit_guard import LadyBanditGuard, make_lady_bandit_guard
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.policies.pointing import PointingPolicy, PointingTarget
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.reference_orbit import mean_motion_host
from orbitalgym.registry import (
    ActionComponentKey,
    AttitudeDynamicsKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
)
from orbitalgym.rollout import belief_rollout
from tests.helpers.minimal_scenario import minimal_scenario_kwargs

_N_STEPS = 12
_DT = 10.0
_HALF_ANGLE_RAD = float(jnp.deg2rad(5.0))
_RING_M = 500.0
_SYN_T = 6
_SYN_MASK = jnp.array([True, True, True, True, False, False])


def _zero_policies(env, cfg):
    return BySide(
        guard=dataclasses.replace(
            ZeroControl(), command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards
        ),
        bandit=dataclasses.replace(
            ZeroControl(), command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits
        ),
    )


def _init_ps_fns():
    return BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)


def _tracking_kf_env():
    """LBG with full observation and a KF whose predict matches the truth STM."""
    cfg = make_lady_bandit_guard(max_horizon_s=_N_STEPS * _DT + _DT, dt=_DT)
    env = OrbitalGymEnv(cfg)
    d = env.layout.dynamics_state_dim
    stm = hcw_rtn_stm(mean_motion_host(cfg.reference_orbit), _DT)
    updater = KFBeliefUpdater(
        stm=stm,
        control_matrix=jnp.zeros((d, 3)),
        process_noise=jnp.eye(d) * 1e-6,
    )
    belief_init = BySide(
        guard=KFFromTruthInitializer(layout=env.layout, variance_diag=jnp.ones(d)),
        bandit=KFFromTruthInitializer(layout=env.layout, variance_diag=jnp.ones(d)),
    )
    return env, cfg, belief_init, BySide(guard=updater, bandit=updater)


def test_full_observation_kf_tracks_the_bandit():
    env, cfg, belief_init, belief_upd = _tracking_kf_env()
    traj, beliefs = belief_rollout(
        env=env,
        policies=_zero_policies(env, cfg),
        init_policy_state_fns=_init_ps_fns(),
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
        key=jax.random.PRNGKey(0),
        n_steps=_N_STEPS,
    )
    m = lbg_episode_metrics(traj, cfg, belief_guard=beliefs.guard, detect_error_m=100.0)

    assert float(m.belief_err_guard) < 1.0
    assert float(m.time_to_detect_guard) == _DT
    assert float(m.time_to_belief_error_below_threshold_guard) == 0.0
    assert float(m.belief_age_guard) == 0.0


def test_commit_error_is_nan_when_no_bandit_commits():
    env, cfg, belief_init, belief_upd = _tracking_kf_env()
    traj, beliefs = belief_rollout(
        env=env,
        policies=_zero_policies(env, cfg),
        init_policy_state_fns=_init_ps_fns(),
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
        key=jax.random.PRNGKey(0),
        n_steps=_N_STEPS,
    )
    never = lbg_episode_metrics(traj, cfg, belief_guard=beliefs.guard, commit_radius_m=1.0)
    always = lbg_episode_metrics(traj, cfg, belief_guard=beliefs.guard, commit_radius_m=1e9)

    assert np.isnan(float(never.belief_err_guard_at_commit))
    assert float(always.belief_err_guard_at_commit) < 1.0


def _cone_pointed_away_cfg():
    """Guard holds its 5-degree cone on +x while the bandit sits at -x."""
    kwargs = minimal_scenario_kwargs(
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
    cfg = ScenarioConfig(**kwargs)
    obs_fn = ConicalObservation(
        layout=cfg.layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]]),
        half_angle_rad=_HALF_ANGLE_RAD,
        sigma_floor=0.1,
    )
    return dataclasses.replace(cfg, guard_observation_fn=obs_fn)


def _pin_geometry(state):
    quat_dtype = state.guards.quat.dtype
    guards = state.guards.replace(
        quat=jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=quat_dtype),
        omega=jnp.zeros((1, 3), dtype=state.guards.omega.dtype),
    )
    new_rt = state.bandits.rt.at[0, :].set(
        jnp.array([-_RING_M, 0.0, 0.0, 0.0], dtype=state.bandits.rt.dtype)
    )
    return state.replace(guards=guards, bandits=state.bandits.replace(rt=new_rt))


def _unseen_ring_rollout():
    cfg = _cone_pointed_away_cfg()
    env = OrbitalGymEnv(cfg)
    d = env.layout.dynamics_state_dim
    ring_prior = jnp.zeros((d,)).at[1].set(_RING_M)
    belief_init = BySide(
        guard=KFUniformDefaultInitializer(
            layout=env.layout, default_mean=ring_prior, variance_diag=jnp.ones(d) * _RING_M**2
        ),
        bandit=KFFromTruthInitializer(layout=env.layout, variance_diag=jnp.ones(d)),
    )
    updater = KFBeliefUpdater(
        stm=jnp.eye(d),
        control_matrix=jnp.zeros((d, 2)),
        process_noise=jnp.eye(d) * 1e-6,
    )
    guard_zero = dataclasses.replace(
        ZeroControl(), command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards
    )
    policies = BySide(
        guard=PointingPolicy(inner=guard_zero, target=PointingTarget.HOLD),
        bandit=dataclasses.replace(
            ZeroControl(), command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits
        ),
    )
    state, _ = env.reset(jax.random.PRNGKey(0))
    state = _pin_geometry(state)
    traj, beliefs = belief_rollout(
        env=env,
        policies=policies,
        init_policy_state_fns=_init_ps_fns(),
        belief_initializers=belief_init,
        belief_updaters=BySide(guard=updater, bandit=updater),
        key=jax.random.PRNGKey(1),
        n_steps=_N_STEPS,
        initial_state=state,
    )
    return traj, beliefs, cfg


def test_ring_prior_with_a_blind_cone_never_detects():
    traj, beliefs, cfg = _unseen_ring_rollout()
    assert not bool(jnp.any(traj.visible.guard))
    m = lbg_episode_metrics(traj, cfg, belief_guard=beliefs.guard, detect_error_m=100.0)

    assert float(m.belief_err_guard) > _RING_M
    assert float(m.belief_err_guard) < 3.0 * _RING_M
    assert np.isnan(float(m.time_to_detect_guard))


def test_belief_age_grows_linearly_while_unseen():
    traj, beliefs, cfg = _unseen_ring_rollout()
    m = lbg_episode_metrics(traj, cfg, belief_guard=beliefs.guard)

    # Age at step t is (t + 1) steps when no guard has ever seen a bandit.
    expected = _DT * float(np.mean(np.arange(1, _N_STEPS + 1)))
    assert float(m.belief_age_guard) == np.float32(expected)


def test_evaluate_bank_reports_information_metrics(make_minimal_lbg_env_with_kf):
    env, belief_init, belief_upd, init_ps_fns, policies = make_minimal_lbg_env_with_kf()
    states = sample_bank(env, n_episodes=3, seed=5)
    metrics = evaluate_bank(
        env,
        env.config,
        states,
        jax.random.PRNGKey(0),
        n_steps=4,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
        commit_radius_m=1e9,
        detect_error_m=1e9,
    )
    assert metrics.belief_err_guard.shape == (3,)
    assert np.all(np.isfinite(np.asarray(metrics.belief_err_guard)))
    assert np.all(np.isfinite(np.asarray(metrics.belief_err_guard_at_commit)))
    assert np.all(np.asarray(metrics.time_to_detect_guard) == env.config.dt)
    assert np.all(np.asarray(metrics.time_to_belief_error_below_threshold_guard) == 0.0)

    rows = metrics_to_records(metrics)
    assert isinstance(rows[0]["belief_err_guard"], float)
    assert isinstance(rows[0]["belief_age_guard"], float)
    assert rows[0]["time_to_detect_guard"] == env.config.dt
    assert rows[0]["time_to_belief_error_below_threshold_guard"] == 0.0


def test_information_metrics_are_nan_without_a_belief_log():
    env, cfg, belief_init, belief_upd = _tracking_kf_env()
    traj, _ = belief_rollout(
        env=env,
        policies=_zero_policies(env, cfg),
        init_policy_state_fns=_init_ps_fns(),
        belief_initializers=belief_init,
        belief_updaters=belief_upd,
        key=jax.random.PRNGKey(0),
        n_steps=4,
    )
    m = lbg_episode_metrics(traj, cfg)
    assert np.isnan(float(m.belief_err_guard))
    assert np.isnan(float(m.belief_age_guard))


def _synthetic_inputs():
    """A 2-guard, 2-bandit episode with hand-set beliefs and a step-4 stop.

    Bandit 0 holds 3000 m from the lady; bandit 1 closes from 4000 m to
    100 m, so the nearest bandit switches at step 2. Guard 0 is 1000 m off
    about both bandits throughout; guard 1 is far off about bandit 0 and
    closes on bandit 1. The last two steps sit past the episode stop.
    """
    n_g, n_b, d = 2, 2, 6
    b0 = jnp.tile(jnp.array([3000.0, 0.0, 0.0]), (_SYN_T, 1))
    b1_r = jnp.array([4000.0, 4000.0, 2000.0, 2000.0, 100.0, 100.0])
    b1 = jnp.stack([b1_r, jnp.zeros(_SYN_T), jnp.zeros(_SYN_T)], axis=-1)
    bandit_pos = jnp.stack([b0, b1], axis=1)  # (T, n_b, 3)
    bandit_rtn = jnp.concatenate([bandit_pos, jnp.zeros_like(bandit_pos)], axis=-1)

    err = jnp.zeros((_SYN_T, n_g, n_b))
    err = err.at[:, 0, :].set(1000.0)
    err = err.at[:, 1, 0].set(5000.0)
    err = err.at[:, 1, 1].set(jnp.array([800.0, 600.0, 400.0, 50.0, 0.0, 0.0]))
    offset = jnp.zeros((_SYN_T, n_g, n_b, 3)).at[..., 1].set(err)
    belief_pos = bandit_pos[:, None, :, :] + offset  # (T, n_g, n_b, 3)

    mean = jnp.zeros((_SYN_T, n_g, n_g + n_b, d))
    mean = mean.at[:, :, n_g:, :3].set(belief_pos)

    visible = jnp.zeros((_SYN_T, n_g), dtype=bool)
    visible = visible.at[jnp.array([1, 4, 5]), 0].set(True)

    traj = SimpleNamespace(
        env_state=SimpleNamespace(
            bandits=SimpleNamespace(rtn=bandit_rtn), t=jnp.arange(_SYN_T) * _DT
        ),
        visible=BySide(guard=visible, bandit=None),
    )
    return traj, SimpleNamespace(mean=mean)


def test_guard_information_on_a_hand_built_two_by_two_episode():
    traj, belief = _synthetic_inputs()
    info = guard_information(
        traj,
        belief,
        _SYN_MASK,
        jnp.asarray(4),
        dt=_DT,
        commit_radius_m=2500.0,
        detect_error_m=100.0,
    )

    # err_min over the live steps is [1000, 1000, 400, 50]; the nearest
    # bandit switches to bandit 1 at step 2, which is also the commit step.
    assert float(info.belief_err) == 612.5
    assert float(info.belief_err_at_commit) == 400.0
    assert float(info.time_to_detect) == 2 * _DT
    assert float(info.time_to_belief_error_below_threshold) == 3 * _DT
    # A guard sees a bandit at step 1 only, so the live ages are [1, 0, 1, 2].
    assert float(info.belief_age) == 10.0


def test_guard_information_ignores_steps_past_the_episode_stop():
    traj, belief = _synthetic_inputs()
    live = guard_information(traj, belief, _SYN_MASK, jnp.asarray(4), _DT, 2500.0, 100.0)
    whole = guard_information(
        traj, belief, jnp.ones((_SYN_T,), dtype=bool), jnp.asarray(_SYN_T), _DT, 2500.0, 100.0
    )
    assert float(live.belief_err) != float(whole.belief_err)
    assert float(live.belief_age) != float(whole.belief_age)
    assert float(live.belief_err_at_commit) == float(whole.belief_err_at_commit)


def test_sensor_detection_is_independent_of_an_accurate_blind_prior():
    traj, beliefs, cfg = _unseen_ring_rollout()
    n_guard = beliefs.guard.mean.shape[1]
    truth = traj.env_state.bandits.rt
    accurate = beliefs.guard.replace(
        mean=beliefs.guard.mean.at[:, :, n_guard:, :].set(truth[:, None])
    )
    m = lbg_episode_metrics(traj, cfg, belief_guard=accurate)
    assert np.isnan(float(m.time_to_detect_guard))
    assert float(m.time_to_belief_error_below_threshold_guard) == 0.0


def test_sensor_detection_uses_poststep_time_without_belief_history():
    traj, _, cfg = _unseen_ring_rollout()
    # Nonzero episode start; first detection is the observation after step 2.
    traj = traj.replace(
        env_state=traj.env_state.replace(t=traj.env_state.t + 500.0),
        final_state=traj.final_state.replace(t=traj.final_state.t + 500.0),
        visible=BySide(guard=jnp.zeros_like(traj.visible.guard).at[2, 0].set(True), bandit=None),
    )
    m = jax.jit(lambda t: lbg_episode_metrics(t, cfg))(traj)
    assert float(m.time_to_detect_guard) == 3 * _DT
    assert np.isnan(float(m.time_to_belief_error_below_threshold_guard))


def test_missing_visibility_never_synthesizes_a_detection():
    traj, beliefs, cfg = _unseen_ring_rollout()
    for visible in (None, BySide(guard=None, bandit=None)):
        m = lbg_episode_metrics(
            traj.replace(visible=visible), cfg, belief_guard=beliefs.guard, detect_error_m=1e9
        )
        assert np.isnan(float(m.time_to_detect_guard))
        assert float(m.time_to_belief_error_below_threshold_guard) == 0.0


def test_sensor_detection_includes_final_transition_but_excludes_padding():
    traj, _, cfg = _unseen_ring_rollout()
    for first_visible, expected in ((3, 4 * _DT), (4, np.nan)):
        shortened = traj.replace(
            episode_done=jnp.arange(_N_STEPS) >= 3,
            visible=BySide(
                guard=jnp.zeros_like(traj.visible.guard).at[first_visible, 0].set(True), bandit=None
            ),
        )
        m = lbg_episode_metrics(shortened, cfg)
        np.testing.assert_allclose(float(m.time_to_detect_guard), expected)
