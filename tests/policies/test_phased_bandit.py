"""PhasedBandit: coast, transfer onto a standoff ring, free hold, then commit."""

import dataclasses
import functools

import flax
import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.belief.pf import ParticleFilterFromTruthInitializer
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide, Side
from orbitalgym.eval.conformance import check_policy_conforms
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.policies.heuristic.phased_bandit import (
    COAST,
    COMMIT,
    HOLD,
    TRANSFER,
    PhasedBandit,
)
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.reference_orbit import mean_motion_host
from orbitalgym.rollout import rollout
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec
from tests.policies._helpers import make_impulsive_maneuver_command_cls

DT = 10.0
N_MOTION = 1.078e-3
START_RING_M = 7500.0
STANDOFF_M = 2000.0
COAST_S = 300.0
HOLD_S = 6000.0
ROLLOUT_STEPS = 1400


@flax.struct.dataclass
class _Belief:
    mean: jax.Array


def _view(bandit_rtn, guard_rtn):
    return _Belief(mean=jnp.stack([bandit_rtn, guard_rtn])[None])  # (1, 2, 6)


def _ring_state(amplitude_m: float, phase_rad: float, mean_motion: float = N_MOTION):
    """The RTN state on the natural 2:1 ellipse, matching ``RelativeEllipse``."""
    cp, sp = np.cos(phase_rad), np.sin(phase_rad)
    return jnp.array(
        [
            -amplitude_m * cp,
            2.0 * amplitude_m * sp,
            0.0,
            mean_motion * amplitude_m * sp,
            2.0 * mean_motion * amplitude_m * cp,
            0.0,
        ]
    )


def _unit_policy(n_vehicles: int = 1, **overrides):
    params = dict(
        mean_motion=N_MOTION,
        dt=DT,
        n_vehicles=n_vehicles,
        n_opponents=1,
        state_dim=6,
        command_cls=make_impulsive_maneuver_command_cls(n_vehicles),
        max_dv_mps=0.4,
        t_coast_s=COAST_S,
        standoff_m=STANDOFF_M,
        t_hold_s=HOLD_S,
    )
    params.update(overrides)
    return PhasedBandit.build(**params)


def test_coast_emits_zero_for_exactly_the_coast_duration():
    policy = _unit_policy()
    view = _view(_ring_state(START_RING_M, 0.3), _ring_state(300.0, 1.1))
    state = policy.init_state()
    coast_steps = int(COAST_S / DT)
    for step in range(coast_steps):
        cmd, state = policy(state, view, jax.random.key(0), jnp.asarray(step * DT))
        assert float(jnp.max(jnp.abs(cmd.dv))) == 0.0, f"thrust during coast at step {step}"
        assert int(state.phase[0]) == (COAST if step < coast_steps - 1 else TRANSFER)
    cmd, state = policy(state, view, jax.random.key(0), jnp.asarray(coast_steps * DT))
    assert float(jnp.linalg.norm(cmd.dv)) > 0.0, "no thrust on the first transfer step"


def test_each_vehicle_advances_its_own_phase():
    """A vehicle already on the ring leaves the transfer while a distant one keeps burning."""
    policy = _unit_policy(n_vehicles=2, n_opponents=0)
    mean = jnp.stack(
        [
            jnp.stack([_ring_state(STANDOFF_M, 0.0), jnp.zeros(6)]),
            jnp.stack([jnp.zeros(6), _ring_state(START_RING_M, jnp.pi)]),
        ]
    )
    start = policy.init_state().replace(phase=jnp.asarray([TRANSFER, TRANSFER], dtype=jnp.int32))
    cmd, state = policy(start, _Belief(mean=mean), jax.random.key(0), jnp.asarray(0.0))
    assert int(state.phase[0]) == HOLD
    assert int(state.phase[1]) == TRANSFER
    assert float(jnp.linalg.norm(cmd.dv[0])) < 1e-6
    assert float(jnp.linalg.norm(cmd.dv[1])) > 0.0


@functools.lru_cache(maxsize=1)
def _phased_rollout():
    """One LBG episode: a phased bandit from a 7.5 km ring against a coasting guard."""
    cfg = make_lady_bandit_guard(
        dt=DT,
        max_horizon_s=DT * ROLLOUT_STEPS + DT,
        breach_radius_m=0.0,
        catch_radius_m=0.0,
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(radial_ellipse_m=START_RING_M, phase_rad=jnp.pi),
        ),
    )
    env = OrbitalGymEnv(cfg)
    cap = cfg.bandit_params.max_thrust_n * cfg.dt / cfg.bandit_params.dry_mass_kg
    policy = PhasedBandit.build(
        mean_motion=mean_motion_host(cfg.reference_orbit),
        dt=cfg.dt,
        n_vehicles=cfg.n_bandits,
        n_opponents=cfg.n_guards,
        state_dim=env.layout.dynamics_state_dim,
        command_cls=env.bandit_command_cls,
        max_dv_mps=cap,
        t_coast_s=COAST_S,
        standoff_m=STANDOFF_M,
        t_hold_s=HOLD_S,
        t_transfer_max_s=4000.0,
    )
    guard = dataclasses.replace(
        ZeroControl(), command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards
    )
    traj = rollout(
        env,
        BySide(guard=guard, bandit=policy),
        BySide(
            guard=lambda config, env_state, key: None,
            bandit=lambda config, env_state, key: policy.init_state(),
        ),
        jax.random.key(0),
        n_steps=ROLLOUT_STEPS,
    )
    dv_norm = np.asarray(jnp.linalg.norm(traj.sides.bandit.action.dv, axis=-1)).reshape(-1)
    rtn = np.asarray(traj.env_state.bandits.rtn)[:, 0, :]
    orbit_steps = int(round(2.0 * np.pi / mean_motion_host(cfg.reference_orbit) / DT))
    return dv_norm, rtn, cap, orbit_steps


def _phase_boundaries(dv_norm: np.ndarray) -> tuple[int, int]:
    """``(last transfer step, last hold step)`` read off the thrust trace."""
    thrusting = np.nonzero(dv_norm > 0.0)[0]
    assert thrusting.size > 0, "the bandit never thrusts"
    gap = np.nonzero(np.diff(thrusting) > 1)[0]
    assert gap.size > 0, "the transfer never ends"
    return int(thrusting[gap[0]]), int(thrusting[gap[0] + 1]) - 1


def test_transfer_settles_on_the_standoff_ring_and_the_hold_is_free():
    dv_norm, rtn, _, orbit_steps = _phased_rollout()
    transfer_end, hold_end = _phase_boundaries(dv_norm)
    assert hold_end - transfer_end >= orbit_steps, "hold shorter than one orbit"

    hold = rtn[transfer_end + 1 : transfer_end + 1 + orbit_steps]
    assert float(np.max(dv_norm[transfer_end + 1 : hold_end + 1])) == 0.0, "fuel spent holding"

    # Radial amplitude of the 2:1 ellipse the vehicle is on: r = -A cos(theta),
    # t = 2A sin(theta), so A = sqrt(r^2 + (t/2)^2) is constant on the ring.
    amplitude = np.sqrt(hold[:, 0] ** 2 + (hold[:, 1] / 2.0) ** 2)
    deviation = np.max(np.abs(amplitude - STANDOFF_M)) / STANDOFF_M
    assert deviation < 0.1, f"ring amplitude wandered {deviation:.1%} over an orbit"


def test_commit_closes_on_the_lady():
    dv_norm, rtn, _, _ = _phased_rollout()
    _, hold_end = _phase_boundaries(dv_norm)
    distance = np.linalg.norm(rtn[hold_end + 1 :, :3], axis=-1)
    speed = np.linalg.norm(rtn[hold_end + 1 :, 3:6], axis=-1)
    arrived = (distance < 5.0) & (speed < 0.5)
    assert arrived.any(), f"closest approach {distance.min():.1f} m at {speed.min():.3f} m/s"


def test_commanded_norm_never_exceeds_the_cap():
    dv_norm, _, cap, _ = _phased_rollout()
    assert float(np.max(dv_norm)) <= cap + 1e-6


def test_conforms_with_belief_views():
    cfg = make_lady_bandit_guard(max_horizon_s=600.0)
    env = OrbitalGymEnv(cfg)
    policy = PhasedBandit.build(
        mean_motion=env.mean_motion,
        dt=cfg.dt,
        n_vehicles=cfg.n_bandits,
        n_opponents=cfg.n_guards,
        state_dim=env.layout.dynamics_state_dim,
        command_cls=env.bandit_command_cls,
        max_dv_mps=0.4,
        t_coast_s=COAST_S,
        standoff_m=STANDOFF_M,
        t_hold_s=HOLD_S,
    )
    init = ParticleFilterFromTruthInitializer(layout=env.layout, n_particles=8)
    report = check_policy_conforms(policy, env, Side.BANDIT, init)
    assert report.command_ok and report.traceable and report.rollout_ok, report.message


def test_flat_observation_matches_the_belief_path():
    policy = _unit_policy()
    view = _view(_ring_state(START_RING_M, 0.7), _ring_state(300.0, 2.0))
    state = policy.init_state().replace(phase=jnp.asarray([TRANSFER], dtype=jnp.int32))
    belief_cmd, _ = policy(state, view, jax.random.key(0), jnp.asarray(0.0))
    flat_cmd, _ = policy(state, view.mean.reshape(-1), jax.random.key(0), jnp.asarray(0.0))
    assert jnp.allclose(belief_cmd.dv, flat_cmd.dv)


def test_commit_matches_the_glideslope_policy_it_wraps():
    policy = _unit_policy(avoidance_gain_mps=0.5)
    view = _view(_ring_state(STANDOFF_M, 0.4), _ring_state(400.0, 1.5))
    state = policy.init_state().replace(phase=jnp.asarray([COMMIT], dtype=jnp.int32))
    cmd, _ = policy(state, view, jax.random.key(0), jnp.asarray(0.0))
    expected, _ = policy.commit(None, view, jax.random.key(0), jnp.asarray(0.0))
    assert jnp.allclose(cmd.dv, expected.dv)
