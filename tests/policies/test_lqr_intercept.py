"""Guard-side LQR intercept: relative-state regulation onto the nearest opponent."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.policies.heuristic.lqr_avoid import LQRGoToLadyWithAvoidance, _hcw_rt_ab
from orbitalgym.policies.heuristic.lqr_intercept import LQRIntercept
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.reference_orbit import mean_motion_host
from orbitalgym.rollout import rollout
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec
from tests.policies._helpers import make_impulsive_maneuver_command_cls

MEAN_MOTION = 0.0010780076263472438
DT = 10.0
ONE_ORBIT_STEPS = 583


def test_closed_loop_in_own_hcw_model_drives_the_relative_state_to_zero():
    cap = 0.455
    a, b = _hcw_rt_ab(MEAN_MOTION, DT)
    gain = np.asarray(
        LQRIntercept.build(
            mean_motion=MEAN_MOTION,
            dt=DT,
            n_vehicles=1,
            n_opponents=1,
            state_dim=6,
            command_cls=make_impulsive_maneuver_command_cls(1),
            max_dv_mps=cap,
        ).gain,
        dtype=np.float64,
    )

    x = np.array([2939.0, 1199.0, 0.65, -6.34])  # relative state, |r| ≈ 3 km
    settled = np.zeros(600, dtype=bool)
    for step in range(600):
        u = -gain @ x
        norm = np.linalg.norm(u)
        if norm > cap:
            u = u * (cap / norm)
        x = a @ x + b @ u
        settled[step] = np.linalg.norm(x[:2]) < 5.0 and np.linalg.norm(x[2:]) < 0.5

    assert settled.any(), "never reached 5 m with relative speed under 0.5 m/s within 600 steps"
    first = int(np.argmax(settled))
    assert settled[first:].all(), f"left the 5 m ball after step {first}"


def _intercept_rollout(bandit_policy_kind: str):
    """One-orbit LBG rollout: intercept guard on a 300 m ring versus a 3 km bandit."""
    cfg = make_lady_bandit_guard(
        dt=DT,
        max_horizon_s=6000.0,
        breach_radius_m=0.0,
        catch_radius_m=0.0,
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=300.0,
                mass_sampler=ConstantMass(propellant_mass_kg=50.0),
            ),
            bandit_sampler=RelativeEllipse(radial_ellipse_m=3000.0, phase_rad=jnp.pi),
        ),
    )
    env = OrbitalGymEnv(cfg)
    mean_motion = mean_motion_host(cfg.reference_orbit)
    guard_cap = cfg.guard_params.max_thrust_n * cfg.dt / cfg.guard_params.dry_mass_kg
    guard_policy = LQRIntercept.build(
        mean_motion=mean_motion,
        dt=cfg.dt,
        n_vehicles=cfg.n_guards,
        n_opponents=cfg.n_bandits,
        state_dim=env.layout.dynamics_state_dim,
        command_cls=env.guard_command_cls,
        max_dv_mps=guard_cap,
    )
    if bandit_policy_kind == "coast":
        bandit_policy = dataclasses.replace(
            ZeroControl(), command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits
        )
    else:
        bandit_policy = LQRGoToLadyWithAvoidance.build(
            mean_motion=mean_motion,
            dt=cfg.dt,
            n_vehicles=cfg.n_bandits,
            n_opponents=cfg.n_guards,
            state_dim=env.layout.dynamics_state_dim,
            command_cls=env.bandit_command_cls,
            max_dv_mps=cfg.bandit_params.max_thrust_n * cfg.dt / cfg.bandit_params.dry_mass_kg,
            avoidance_gain=0.0,
        )
    traj = rollout(
        env,
        BySide(guard=guard_policy, bandit=bandit_policy),
        BySide(
            guard=lambda config, env_state, key: None,
            bandit=lambda config, env_state, key: None,
        ),
        jax.random.key(0),
        n_steps=ONE_ORBIT_STEPS,
    )
    separation = jnp.linalg.norm(
        traj.env_state.guards.rtn[:, :, :3] - traj.env_state.bandits.rtn[:, :, :3], axis=-1
    ).reshape(-1)
    return traj, separation, guard_cap


def test_intercepts_a_coasting_bandit_within_one_orbit():
    _, separation, _ = _intercept_rollout("coast")
    closest_m = float(jnp.min(separation))
    assert closest_m < 5.0, f"closest approach {closest_m:.3f} m"


def test_closes_on_an_lqr_bandit_heading_for_the_lady():
    """The bandit and guard both converge on the lady, so a whole-orbit minimum
    separation is trivially small. The guard cannot outrun a bandit that closes
    3 km in 730 s from a 300 m ring, so this checks two properties instead of a
    single-episode intercept-before-arrival: the guard's closest approach over
    the steps before the bandit itself first comes within 50 m of the lady is
    the measured pursuit capability at this geometry (203 m in float64, well
    under the 250 m bound); and once the bandit has passed the lady, the guard
    still rendezvouses with it, holding under 5 m at under 0.5 m/s relative
    speed for the trajectory's last 50 steps.
    """
    traj, separation, _ = _intercept_rollout("lqr")
    lady_dist = jnp.linalg.norm(traj.env_state.bandits.rtn[:, :, :3], axis=-1).reshape(-1)
    within_50m = lady_dist < 50.0
    cutoff = int(jnp.argmax(within_50m)) if bool(jnp.any(within_50m)) else separation.shape[0]
    assert cutoff > 0, "bandit starts within 50 m of the lady"
    closest_m = float(jnp.min(separation[:cutoff]))
    assert closest_m < 250.0, f"closest approach before breach {closest_m:.3f} m"

    relative_speed = jnp.linalg.norm(
        traj.env_state.guards.rtn[:, :, 3:6] - traj.env_state.bandits.rtn[:, :, 3:6], axis=-1
    ).reshape(-1)
    rendezvoused = (separation < 5.0) & (relative_speed < 0.5)
    assert bool(jnp.all(rendezvoused[-50:])), "guard did not hold rendezvous for the final 50 steps"


def test_commanded_norm_never_exceeds_the_cap():
    traj, _, cap = _intercept_rollout("coast")
    dv_norm = jnp.linalg.norm(traj.sides.guard.action.dv, axis=-1)
    assert float(jnp.max(dv_norm)) <= cap + 1e-6
