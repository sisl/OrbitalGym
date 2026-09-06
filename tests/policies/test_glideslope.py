"""Glideslope guidance: approach, intercept, and cap-independence of the gain."""

import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.policies.heuristic.glideslope import GlideslopeIntercept, GlideslopeToLady
from tests.policies._helpers import make_impulsive_maneuver_command_cls

MEAN_MOTION = 0.0010780076263472438
DT = 10.0
GUARD_CAP = 0.4545
SLOPE_S = 300.0
ARRIVAL_MPS = 0.3
BRAKE_FRACTION = 0.5


def _hcw_rt_ab(mean_motion: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete in-plane HCW (A, B) for state [r, t, r_dot, t_dot], control [dv_r, dv_t]."""
    n = float(mean_motion)
    s = np.sin(n * dt)
    c = np.cos(n * dt)
    phi = np.array(
        [
            [4 - 3 * c, 0.0, s / n, 2 * (1 - c) / n],
            [6 * (s - n * dt), 1.0, -2 * (1 - c) / n, (4 * s - 3 * n * dt) / n],
            [3 * n * s, 0.0, c, 2 * s],
            [-6 * n * (1 - c), 0.0, -2 * s, 4 * c - 3],
        ],
        dtype=np.float64,
    )
    return phi, phi[:, 2:]


A_HCW, _ = _hcw_rt_ab(MEAN_MOTION, DT)


def _commanded_speed(rho: float, cap: float) -> float:
    """The glideslope's commanded closing speed at range `rho` for a vehicle with cap `cap`."""
    a_brake = BRAKE_FRACTION * cap / DT
    return float(min(rho / SLOPE_S + ARRIVAL_MPS, np.sqrt(2.0 * a_brake * rho)))


def _ellipse_state(amplitude_m: float, phase_rad: float) -> np.ndarray:
    """In-plane state on the drift-free 2:1 HCW ellipse of the given radial amplitude."""
    n = MEAN_MOTION
    s = np.sin(phase_rad)
    c = np.cos(phase_rad)
    return np.array(
        [amplitude_m * s, 2.0 * amplitude_m * c, amplitude_m * n * c, -2.0 * amplitude_m * n * s]
    )


def _view(own_rt: np.ndarray, opp_rt: np.ndarray) -> jax.Array:
    """Flat one-vehicle, one-opponent full-state observation from two in-plane states."""
    mean = np.zeros((1, 2, 6))
    for slot, x in ((0, own_rt), (1, opp_rt)):
        mean[0, slot, 0] = x[0]
        mean[0, slot, 1] = x[1]
        mean[0, slot, 3] = x[2]
        mean[0, slot, 4] = x[3]
    return jnp.asarray(mean.reshape(-1), dtype=jnp.float32)


def _to_lady(cap: float, avoidance_gain_mps: float = 0.0) -> GlideslopeToLady:
    return GlideslopeToLady.build(
        mean_motion=MEAN_MOTION,
        dt=DT,
        n_vehicles=1,
        n_opponents=1,
        state_dim=6,
        command_cls=make_impulsive_maneuver_command_cls(1),
        max_dv_mps=cap,
        avoidance_gain_mps=avoidance_gain_mps,
    )


def _command(policy, own_rt: np.ndarray, opp_rt: np.ndarray) -> np.ndarray:
    action, _ = policy(None, _view(own_rt, opp_rt), jax.random.key(0), jnp.asarray(0))
    return np.asarray(action.dv[0, :2], dtype=np.float64)


def test_free_space_approach_arrives_slowly():
    """Post-impulse speed tracks the commanded closing speed once the cap stops binding.

    While the impulse is clipped the vehicle cannot be on the glideslope, and
    from a 6 km line-of-sight range the braking curve lets it build a speed the
    glideslope then asks it to shed faster than half the budget allows. The
    speed bound is therefore checked from the first unsaturated command onward.
    """
    policy = _to_lady(GUARD_CAP)
    far_opponent = np.array([0.0, 5.0e5, 0.0, 0.0])
    starts = [np.array([3000.0, 0.0, 0.0, 0.0])] + [
        _ellipse_state(3000.0, phase) for phase in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    ]

    for start in starts:
        x = start.copy()
        arrival_step = None
        arrival_speed = None
        max_norm = 0.0
        unsaturated_step = None
        speed_excess = -np.inf
        for step in range(3000):
            dv = _command(policy, x, far_opponent)
            dv_norm = float(np.linalg.norm(dv))
            max_norm = max(max_norm, dv_norm)
            rho = float(np.linalg.norm(x[:2]))
            post = x.copy()
            post[2:] += dv
            if unsaturated_step is None and step >= 20 and dv_norm < GUARD_CAP - 1e-6:
                unsaturated_step = step
            if unsaturated_step is not None:
                excess = float(np.linalg.norm(post[2:])) - _commanded_speed(rho, GUARD_CAP)
                speed_excess = max(speed_excess, excess)
            if arrival_step is None and rho < 5.0:
                arrival_step = step
                arrival_speed = float(np.linalg.norm(x[2:]))
            x = A_HCW @ post

        assert arrival_step is not None, f"never reached 5 m from {start}"
        assert arrival_speed < 0.5, f"arrived at {arrival_speed:.3f} m/s from {start}"
        assert unsaturated_step is not None and unsaturated_step <= 60, (
            f"still thrust-limited after {unsaturated_step} steps from {start}"
        )
        assert speed_excess <= 0.05, f"speed exceeded the glideslope by {speed_excess:.4f} m/s"
        assert max_norm <= GUARD_CAP + 1e-6, f"commanded {max_norm:.6f} m/s over cap {GUARD_CAP}"


def test_gain_independent_of_cap():
    x = np.array([60.0, 0.0, -0.2, 0.0])
    far_opponent = np.array([0.0, 5.0e5, 0.0, 0.0])
    slow = _command(_to_lady(GUARD_CAP), x, far_opponent)
    fast = _command(_to_lady(2.0 * GUARD_CAP), x, far_opponent)
    assert np.linalg.norm(slow) < GUARD_CAP - 1e-3, "test state is saturated"
    np.testing.assert_allclose(slow, fast, atol=1e-6)


def test_cap_only_raises_the_clip():
    x = np.array([3000.0, 0.0, 0.0, 0.0])
    far_opponent = np.array([0.0, 5.0e5, 0.0, 0.0])
    for cap in (GUARD_CAP, 2.0 * GUARD_CAP):
        norm = float(np.linalg.norm(_command(_to_lady(cap), x, far_opponent)))
        assert abs(norm - cap) < 1e-5, f"first command norm {norm:.6f} at cap {cap}"


def test_intercept_closes_on_moving_target():
    policy = GlideslopeIntercept.build(
        mean_motion=MEAN_MOTION,
        dt=DT,
        n_vehicles=1,
        n_opponents=1,
        state_dim=6,
        command_cls=make_impulsive_maneuver_command_cls(1),
        max_dv_mps=GUARD_CAP,
    )
    x = _ellipse_state(300.0, 0.5 * np.pi)
    target = _ellipse_state(3000.0, 0.0)
    closed = None
    for step in range(1200):
        dv = _command(policy, x, target)
        assert np.linalg.norm(dv) <= GUARD_CAP + 1e-6
        rel_range = float(np.linalg.norm(x[:2] - target[:2]))
        rel_speed = float(np.linalg.norm(x[2:] - target[2:]))
        if closed is None and rel_range < 20.0 and rel_speed < 0.5:
            closed = (step, rel_range, rel_speed)
        post = x.copy()
        post[2:] += dv
        x = A_HCW @ post
        target = A_HCW @ target
    assert closed is not None, "guard never closed to 20 m under 0.5 m/s within 1200 steps"


def test_runs_under_jit_and_vmap():
    n_vehicles, n_opponents, state_dim = 2, 1, 6
    command_cls = make_impulsive_maneuver_command_cls(n_vehicles)
    policy = GlideslopeToLady.build(
        mean_motion=MEAN_MOTION,
        dt=DT,
        n_vehicles=n_vehicles,
        n_opponents=n_opponents,
        state_dim=state_dim,
        command_cls=command_cls,
        max_dv_mps=GUARD_CAP,
        avoidance_gain_mps=0.4545,
    )
    key = jax.random.key(0)
    views = jax.random.normal(key, (3, n_vehicles * (n_vehicles + n_opponents) * state_dim)) * 100.0

    def run(view):
        action, _ = policy(None, view, key, jnp.asarray(0))
        return action.dv

    dv = jax.jit(jax.vmap(run))(views)
    dv_dim = command_cls.zeros(n_vehicles).dv.shape[-1]
    assert dv.shape == (3, n_vehicles, dv_dim)
    assert bool(jnp.all(jnp.isfinite(dv)))
