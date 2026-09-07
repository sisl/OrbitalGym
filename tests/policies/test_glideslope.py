"""Glideslope guidance: approach, intercept, sustainable speed, and gain independence."""

import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.policies.heuristic.glideslope import (
    GlideslopeIntercept,
    GlideslopeToLady,
    glideslope_u,
)
from tests.policies._helpers import make_impulsive_maneuver_command_cls

MEAN_MOTION = 0.0010780076263472438
DT = 10.0
GUARD_CAP = 0.4545
SLOPE_S = 100.0
ARRIVAL_MPS = 0.3
BRAKE_FRACTION = 0.5
HCW_FRACTION = 0.5
S_HCW = HCW_FRACTION * GUARD_CAP / (2.0 * MEAN_MOTION * DT)


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


def _commanded_speed(rho: float, cap: float, hcw_fraction: float = HCW_FRACTION) -> float:
    """The glideslope's commanded closing speed at range `rho` for a vehicle with cap `cap`."""
    a_brake = BRAKE_FRACTION * cap / DT
    s_hcw = hcw_fraction * cap / (2.0 * MEAN_MOTION * DT)
    return float(min(rho / SLOPE_S + ARRIVAL_MPS, np.sqrt(2.0 * a_brake * rho), s_hcw))


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


def _to_lady(
    cap: float,
    avoidance_gain_mps: float = 0.0,
    hcw_fraction: float = HCW_FRACTION,
) -> GlideslopeToLady:
    return GlideslopeToLady.build(
        mean_motion=MEAN_MOTION,
        dt=DT,
        n_vehicles=1,
        n_opponents=1,
        state_dim=6,
        command_cls=make_impulsive_maneuver_command_cls(1),
        max_dv_mps=cap,
        hcw_fraction=hcw_fraction,
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
    speed bound is therefore checked from the first unsaturated command onward,
    and the commands are separately required to stay off the cap for the rest
    of the approach, so the bound is checked on a vehicle that can actually
    reach the commanded velocity. Past arrival the cap binds again as the
    vehicle holds station on the lady, and the speed bound still holds there.
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
        approach_norm = 0.0
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
                if arrival_step is None:
                    approach_norm = max(approach_norm, dv_norm)
            if arrival_step is None and rho < 5.0:
                arrival_step = step
                arrival_speed = float(np.linalg.norm(x[2:]))
            x = A_HCW @ post

        assert arrival_step is not None, f"never reached 5 m from {start}"
        assert arrival_speed < 0.5, f"arrived at {arrival_speed:.3f} m/s from {start}"
        assert unsaturated_step is not None and unsaturated_step <= 60, (
            f"still thrust-limited after {unsaturated_step} steps from {start}"
        )
        assert approach_norm < GUARD_CAP - 1e-6, (
            f"cap bound again during the approach: commanded {approach_norm:.6f} m/s "
            f"at cap {GUARD_CAP} between steps {unsaturated_step} and {arrival_step}"
        )
        assert speed_excess <= 0.05, f"speed exceeded the glideslope by {speed_excess:.4f} m/s"
        assert max_norm <= GUARD_CAP + 1e-6, f"commanded {max_norm:.6f} m/s over cap {GUARD_CAP}"


def test_gain_independent_of_cap():
    # 60 m out and already closing at 0.7 m/s: the glideslope is the smaller of
    # the two speeds and the residual impulse is inside both caps, so neither
    # the braking curve nor the clip can bind for either vehicle.
    x = np.array([60.0, 0.0, -0.7, 0.0])
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


def test_long_range_approach_stays_on_the_line_of_sight():
    """From 20 km the sustainable speed keeps the approach inside the vehicle's budget.

    Holding a relative velocity that is not natural motion costs about
    ``2 n v dt`` of impulse per step, so at the guard cap a straight approach
    can be held only up to ``S_HCW``. Asking for more than the budget can hold
    lets the Coriolis coupling carry the vehicle off the line of sight, and the
    range grows instead of closing. The approach is started on the drift-free
    ring so that the initial state is one the dynamics would sustain by itself.
    """
    policy = _to_lady(GUARD_CAP)
    far_opponent = np.array([0.0, 5.0e5, 0.0, 0.0])
    starts = [
        np.array([0.0, 20000.0, 0.0, 0.0]),
        np.array([-1000.0, 20000.0, 0.0, 2.0 * MEAN_MOTION * 1000.0]),
        np.array([0.0, 22000.0, 1000.0 * MEAN_MOTION, 0.0]),
    ]

    for start in starts:
        x = start.copy()
        arrival_step = None
        max_rho = 0.0
        max_speed = 0.0
        for step in range(4000):
            dv = _command(policy, x, far_opponent)
            rho = float(np.linalg.norm(x[:2]))
            max_rho = max(max_rho, rho)
            post = x.copy()
            post[2:] += dv
            max_speed = max(max_speed, float(np.linalg.norm(post[2:])))
            if arrival_step is None and rho < 5.0:
                arrival_step = step
                break
            x = A_HCW @ post

        assert arrival_step is not None, f"never reached 5 m from {start}"
        assert max_rho < 25000.0, f"drifted out to {max_rho:.0f} m from {start}"
        assert max_speed <= S_HCW + 0.05, (
            f"commanded {max_speed:.3f} m/s over the sustainable {S_HCW:.3f} m/s from {start}"
        )


def test_sustainable_speed_binds_at_three_kilometres():
    """At 3 km the sustainable speed is the smallest of the three terms at the default fraction.

    Both states are already closing near their own commanded speed, so the
    residual impulse is inside the cap and the post-impulse velocity is exactly
    the commanded one.
    """
    far_opponent = np.array([0.0, 5.0e5, 0.0, 0.0])
    a_brake = BRAKE_FRACTION * GUARD_CAP / DT
    s_brake = float(np.sqrt(2.0 * a_brake * 3000.0))
    assert s_brake > S_HCW, "the sustainable speed does not bind at 3 km"

    loose = np.array([3000.0, 0.0, -s_brake - 0.2, 0.0])
    dv = _command(_to_lady(GUARD_CAP, hcw_fraction=10.0), loose, far_opponent)
    assert np.linalg.norm(dv) < GUARD_CAP - 1e-3, "test state is saturated"
    np.testing.assert_allclose(loose[2:] + dv, [-s_brake, 0.0], atol=1e-4)

    tight = np.array([3000.0, 0.0, -S_HCW - 0.2, 0.0])
    dv = _command(_to_lady(GUARD_CAP), tight, far_opponent)
    assert np.linalg.norm(dv) < GUARD_CAP - 1e-3, "test state is saturated"
    np.testing.assert_allclose(tight[2:] + dv, [-S_HCW, 0.0], atol=1e-4)


def _closing_speed_commanded(
    rho_m: float,
    *,
    cap_mps: float,
    rate_rad_s: float,
    step_s: float,
    fraction: float,
) -> float:
    """The closing speed the unclipped law commands from a state at rest at `rho_m`.

    At rest the desired velocity is the closing speed along the line of sight
    and nothing else, so the unclipped impulse is that speed directed inward.
    Reading it before the norm clip keeps the measurement independent of the
    cap, which the clip is tested against separately.
    """
    x = np.array([[rho_m, 0.0, 0.0, 0.0]])
    u = np.asarray(
        glideslope_u(
            jnp.asarray(x),
            jnp.zeros((1, 4)),
            max_dv_mps=cap_mps,
            dt=step_s,
            mean_motion=rate_rad_s,
            slope_s=SLOPE_S,
            arrival_mps=ARRIVAL_MPS,
            brake_fraction=BRAKE_FRACTION,
            hcw_fraction=fraction,
        )[0],
        dtype=np.float64,
    )
    assert abs(u[1]) < 1e-9, "commanded impulse left the line of sight"
    return -u[0]


def test_sustainable_speed_scales_inversely_with_rate_and_step():
    """The sustainable speed goes as ``fraction * cap / (2 * rate * step)``.

    Both the orbital rate and the step length are varied independently, and the
    expectation is written out here from the impulse budget rather than read
    back from the implementation, so a change to either the constant or the
    dependence on ``mean_motion`` and ``dt`` shows up as a failure.
    """
    rho_m = 8000.0
    cap_mps = GUARD_CAP
    measured = {}

    for rate_rad_s in (MEAN_MOTION, 2.0 * MEAN_MOTION):
        for step_s in (DT, 2.0 * DT):
            for fraction in (0.25, 0.5):
                expected = fraction * cap_mps / (2.0 * rate_rad_s * step_s)
                a_brake = BRAKE_FRACTION * cap_mps / step_s
                assert expected < np.sqrt(2.0 * a_brake * rho_m), (
                    f"the braking curve binds at rate={rate_rad_s}, step={step_s}"
                )
                assert expected < rho_m / SLOPE_S + ARRIVAL_MPS, (
                    f"the linear glideslope binds at rate={rate_rad_s}, step={step_s}"
                )
                got = _closing_speed_commanded(
                    rho_m,
                    cap_mps=cap_mps,
                    rate_rad_s=rate_rad_s,
                    step_s=step_s,
                    fraction=fraction,
                )
                assert abs(got - expected) < 1e-3, (
                    f"commanded {got:.4f} m/s, expected {expected:.4f} m/s at "
                    f"rate={rate_rad_s}, step={step_s}, fraction={fraction}"
                )
                measured[(rate_rad_s, step_s, fraction)] = got

    base = measured[(MEAN_MOTION, DT, 0.5)]
    assert abs(measured[(2.0 * MEAN_MOTION, DT, 0.5)] / base - 0.5) < 1e-3, "rate does not halve it"
    assert abs(measured[(MEAN_MOTION, 2.0 * DT, 0.5)] / base - 0.5) < 1e-3, "step does not halve it"
    assert abs(measured[(MEAN_MOTION, DT, 0.25)] / base - 0.5) < 1e-3, "fraction is not linear"
