"""LQR-to-lady with guard avoidance, driven by a belief mean or a flat observation."""

import dataclasses

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.policies.heuristic.lqr_avoid import LQRGoToLadyWithAvoidance, _hcw_rt_ab
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.reference_orbit import mean_motion_host
from orbitalgym.rollout import rollout
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec
from tests.policies._helpers import make_impulsive_maneuver_command_cls

N_MOTION = 1.078e-3


@flax.struct.dataclass
class _Belief:
    mean: jax.Array


def _view(bandit_rtn, guard_rtn):
    rows = jnp.stack([bandit_rtn, guard_rtn])  # own first, then opposing
    return _Belief(mean=rows[None])  # (1, 2, 6)


def _policy(avoidance_gain, max_dv_mps: float = 1.0):
    return LQRGoToLadyWithAvoidance.build(
        mean_motion=N_MOTION,
        dt=10.0,
        n_vehicles=1,
        n_opponents=1,
        state_dim=6,
        command_cls=make_impulsive_maneuver_command_cls(1),
        max_dv_mps=max_dv_mps,
        avoidance_gain=avoidance_gain,
        avoidance_sigma_m=300.0,
    )


BANDIT = jnp.array([500.0, 0.0, 0.0, 0.0, 0.0, 0.0])
GUARD = jnp.array([600.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_pure_lqr_moves_toward_lady():
    cmd, _ = _policy(0.0)(None, _view(BANDIT, GUARD), None, 0.0)
    assert cmd.dv.shape == (1, 3)
    assert cmd.dv[0, 0] < 0.0  # radial component points toward the origin


def test_avoidance_pushes_away_from_guard():
    pure, _ = _policy(0.0, max_dv_mps=100.0)(None, _view(BANDIT, GUARD), None, 0.0)
    avoid, _ = _policy(1.0, max_dv_mps=100.0)(None, _view(BANDIT, GUARD), None, 0.0)
    assert avoid.dv[0, 0] < pure.dv[0, 0]  # guard sits at +R; repulsion adds -R


def test_cross_track_is_zero_and_norm_clip_holds():
    cmd, _ = _policy(5.0)(None, _view(BANDIT, GUARD), None, 0.0)
    assert cmd.dv[0, 2] == 0.0
    assert float(jnp.linalg.norm(cmd.dv[0])) <= 1.0 + 1e-6


def test_flat_observation_matches_belief_path():
    view = _view(BANDIT, GUARD)
    flat = view.mean.reshape(-1)
    a, _ = _policy(1.0)(None, view, None, 0.0)
    b, _ = _policy(1.0)(None, flat, None, 0.0)
    assert jnp.allclose(a.dv, b.dv)


def test_rejects_wrong_sized_flat_observation():
    policy = _policy(0.0)
    wrong_size_flat = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="full-state"):
        policy(None, wrong_size_flat, None, 0.0)


def test_uses_last_channel_of_a_composite_flat_observation():
    """Two equal-sized full-state channels concatenated (as CompositeObservation
    would produce): the trailing channel is used, matching the single-channel
    flat-observation result."""
    view = _view(BANDIT, GUARD)
    single_flat = view.mean.reshape(-1)
    noise_channel = single_flat + 1000.0  # a differing leading channel
    composite_flat = jnp.concatenate([noise_channel, single_flat])
    a, _ = _policy(1.0)(None, single_flat, None, 0.0)
    b, _ = _policy(1.0)(None, composite_flat, None, 0.0)
    assert jnp.allclose(a.dv, b.dv)


def test_closed_loop_in_own_hcw_model_converges_to_the_origin():
    mean_motion = 0.0010780076263472438
    dt = 10.0
    cap = 0.455
    a, b = _hcw_rt_ab(mean_motion, dt)
    gain = np.asarray(
        LQRGoToLadyWithAvoidance.build(
            mean_motion=mean_motion,
            dt=dt,
            n_vehicles=1,
            n_opponents=1,
            state_dim=6,
            command_cls=make_impulsive_maneuver_command_cls(1),
            max_dv_mps=cap,
        ).gain,
        dtype=np.float64,
    )

    x = np.array([2939.0, 1199.0, 0.65, -6.34])
    settled = np.zeros(600, dtype=bool)
    for step in range(600):
        u = -gain @ x
        norm = np.linalg.norm(u)
        if norm > cap:
            u = u * (cap / norm)
        x = a @ x + b @ u
        settled[step] = np.linalg.norm(x[:2]) < 5.0 and np.linalg.norm(x[2:]) < 0.5

    assert settled.any(), "never reached 5 m with speed under 0.5 m/s within 600 steps"
    first = int(np.argmax(settled))
    assert settled[first:].all(), f"left the 5 m ball after step {first}"


def _lqr_bandit_rollout(avoidance_gain: float):
    """One-orbit LBG rollout: LQR bandit versus a drifting guard on a 1 m ring."""
    cfg = make_lady_bandit_guard(
        dt=10.0,
        max_horizon_s=6000.0,
        breach_radius_m=0.0,
        catch_radius_m=0.0,
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(radial_ellipse_m=1000.0, phase_rad=jnp.pi),
        ),
    )
    env = OrbitalGymEnv(cfg)
    cap = cfg.bandit_params.max_thrust_n * cfg.dt / cfg.bandit_params.dry_mass_kg
    bandit_policy = LQRGoToLadyWithAvoidance.build(
        mean_motion=mean_motion_host(cfg.reference_orbit),
        dt=cfg.dt,
        n_vehicles=cfg.n_bandits,
        n_opponents=cfg.n_guards,
        state_dim=env.layout.dynamics_state_dim,
        command_cls=env.bandit_command_cls,
        max_dv_mps=cap,
        avoidance_gain=avoidance_gain,
    )
    guard_policy = dataclasses.replace(
        ZeroControl(), command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards
    )
    traj = rollout(
        env,
        BySide(guard=guard_policy, bandit=bandit_policy),
        BySide(
            guard=lambda config, env_state, key: None,
            bandit=lambda config, env_state, key: None,
        ),
        jax.random.key(0),
        n_steps=583,
    )
    return traj, cap


def test_env_rollout_closes_on_the_lady():
    traj, _ = _lqr_bandit_rollout(avoidance_gain=0.0)
    bandit_pos = traj.env_state.bandits.rtn[:, :, :3]
    assert float(jnp.min(jnp.linalg.norm(bandit_pos, axis=-1))) < 5.0


def test_commanded_norm_never_exceeds_the_cap():
    traj, cap = _lqr_bandit_rollout(avoidance_gain=0.0)
    dv_norm = jnp.linalg.norm(traj.sides.bandit.action.dv, axis=-1)
    assert float(jnp.max(dv_norm)) <= cap + 1e-6
