"""PD and LQR feedback: the laws, the regulator gain, and the policy wrappers."""

import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.policies.heuristic.feedback import (
    LQRFeedback,
    PDFeedback,
    hcw_lqr_gain,
    lqr_impulse,
    pd_impulse,
)
from orbitalgym.registry import PolicyKey, resolve
from tests.policies._helpers import make_impulsive_maneuver_command_cls

MEAN_MOTION = 0.0010780076263472438
DT = 10.0
CAP = 0.1


def _transition() -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(hcw_rtn_stm(MEAN_MOTION, DT), dtype=np.float64)
    return a, a[:, 3:]


def _view(own: np.ndarray, *opponents: np.ndarray) -> jax.Array:
    """Flat one-vehicle full-state observation: own state, then each opponent's."""
    return jnp.asarray(np.concatenate([own, *opponents]), dtype=jnp.float32)


def _regulation_cost(gain: np.ndarray, x0: np.ndarray, steps: int = 4000) -> float:
    """Quadratic cost of ``u = -K x`` from ``x0`` under the default weights."""
    a, b = _transition()
    q = np.diag([1 / 100.0**2] * 3 + [1.0] * 3)
    r = np.eye(3) / 0.1**2
    x, cost = x0.astype(np.float64), 0.0
    for _ in range(steps):
        u = -gain @ x
        cost += x @ q @ x + u @ r @ u
        x = a @ x + b @ u
    return cost


def test_pd_impulse_matches_the_law_and_broadcasts():
    relative = jnp.asarray([[300.0, -40.0, 12.0, 0.5, -0.2, 0.1], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    u = pd_impulse(relative, DT, kp=2e-4, kd=0.03)
    expected = -DT * (2e-4 * np.asarray(relative[:, :3]) + 0.03 * np.asarray(relative[:, 3:]))
    np.testing.assert_allclose(np.asarray(u), expected, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(pd_impulse(relative[0], DT, 2e-4, 0.03)), expected[0], rtol=1e-6
    )


def test_pd_default_gains_are_critically_damped_with_100_s_time_constant():
    kp, kd = 1e-4, 0.02
    natural_frequency = np.sqrt(kp)
    assert np.isclose(kd / (2.0 * natural_frequency), 1.0)
    assert np.isclose(1.0 / natural_frequency, 100.0)
    np.testing.assert_allclose(
        np.asarray(pd_impulse(jnp.ones(6), DT)), np.asarray(pd_impulse(jnp.ones(6), DT, kp, kd))
    )


def test_pd_feedback_regulates_the_hcw_relative_state():
    a, b = _transition()
    x = np.array([200.0, -150.0, 80.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        x = a @ x + b @ np.asarray(pd_impulse(jnp.asarray(x), DT), dtype=np.float64)
    assert np.linalg.norm(x[:3]) < 1.0
    assert np.linalg.norm(x[3:]) < 0.01


def test_lqr_gain_stabilizes_and_is_locally_optimal():
    gain = np.asarray(hcw_lqr_gain(MEAN_MOTION, DT), dtype=np.float64)
    assert gain.shape == (3, 6)
    a, b = _transition()
    assert np.max(np.abs(np.linalg.eigvals(a - b @ gain))) < 1.0
    x0 = np.array([300.0, 100.0, -50.0, 0.1, -0.2, 0.05])
    best = _regulation_cost(gain, x0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        perturbed = gain * (1.0 + 0.05 * rng.standard_normal(gain.shape))
        assert _regulation_cost(perturbed, x0) > best


def test_lqr_gain_follows_the_weight_scales():
    cheap = np.asarray(hcw_lqr_gain(MEAN_MOTION, DT, impulse_scale_mps=1.0))
    default = np.asarray(hcw_lqr_gain(MEAN_MOTION, DT))
    assert np.linalg.norm(cheap) > np.linalg.norm(default)


def test_lqr_impulse_is_minus_gain_times_state_and_broadcasts():
    gain = hcw_lqr_gain(MEAN_MOTION, DT)
    relative = jnp.asarray([[300.0, -40.0, 12.0, 0.5, -0.2, 0.1], [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    expected = -(np.asarray(relative) @ np.asarray(gain).T)
    np.testing.assert_allclose(np.asarray(lqr_impulse(gain, relative)), expected, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(lqr_impulse(gain, relative[1])), expected[1], rtol=1e-5)


def test_pd_policy_commands_the_clipped_law_toward_the_nearest_opponent():
    command_cls = make_impulsive_maneuver_command_cls(1)
    policy = PDFeedback.build(
        dt=DT, n_vehicles=1, n_opponents=2, command_cls=command_cls, max_dv_mps=CAP
    )
    own = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    far = np.array([900.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    near = np.array([10.0, 40.0, 0.0, 0.0, 0.02, 0.0])
    command, state = policy(None, _view(own, far, near), jax.random.PRNGKey(0), jnp.array(0.0))
    expected = np.asarray(pd_impulse(jnp.asarray(own - near), DT))
    assert state is None
    assert np.linalg.norm(expected) < CAP
    np.testing.assert_allclose(np.asarray(command.dv[0]), expected, rtol=1e-5, atol=1e-8)


def test_pd_policy_limits_the_impulse_and_keeps_its_direction():
    command_cls = make_impulsive_maneuver_command_cls(1)
    policy = PDFeedback.build(
        dt=DT, n_vehicles=1, n_opponents=1, command_cls=command_cls, max_dv_mps=CAP
    )
    own = np.array([3000.0, -4000.0, 0.0, 0.0, 0.0, 0.0])
    command, _ = policy(None, _view(own, np.zeros(6)), jax.random.PRNGKey(0), jnp.array(0.0))
    dv = np.asarray(command.dv[0])
    assert np.isclose(np.linalg.norm(dv), CAP, rtol=1e-5)
    np.testing.assert_allclose(dv / CAP, [-0.6, 0.8, 0.0], atol=1e-5)


def test_policies_regulate_to_the_origin_when_asked():
    command_cls = make_impulsive_maneuver_command_cls(1)
    own = np.array([20.0, -10.0, 5.0, 0.01, 0.0, -0.02])
    opponent = np.array([500.0, 500.0, 0.0, 0.0, 0.0, 0.0])
    view = _view(own, opponent)
    key, t = jax.random.PRNGKey(0), jnp.array(0.0)
    pd = PDFeedback.build(
        dt=DT, n_vehicles=1, n_opponents=1, command_cls=command_cls, max_dv_mps=CAP, to_origin=True
    )
    lqr = LQRFeedback.build(
        mean_motion=MEAN_MOTION,
        dt=DT,
        n_vehicles=1,
        n_opponents=1,
        command_cls=command_cls,
        max_dv_mps=CAP,
        to_origin=True,
    )
    gain = hcw_lqr_gain(MEAN_MOTION, DT)
    np.testing.assert_allclose(
        np.asarray(pd(None, view, key, t)[0].dv[0]),
        np.asarray(pd_impulse(jnp.asarray(own), DT)),
        rtol=1e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(lqr(None, view, key, t)[0].dv[0]),
        np.asarray(lqr_impulse(gain, jnp.asarray(own, dtype=jnp.float32))),
        rtol=1e-4,
        atol=1e-7,
    )


def test_lqr_policy_closes_on_a_coasting_opponent():
    command_cls = make_impulsive_maneuver_command_cls(1)
    policy = LQRFeedback.build(
        mean_motion=MEAN_MOTION,
        dt=DT,
        n_vehicles=1,
        n_opponents=1,
        command_cls=command_cls,
        max_dv_mps=CAP,
    )
    a, b = _transition()
    own = np.array([150.0, 200.0, -100.0, 0.0, 0.0, 0.0])
    opponent = np.zeros(6)
    key, t = jax.random.PRNGKey(0), jnp.array(0.0)
    for _ in range(300):
        command, _ = policy(None, _view(own, opponent), key, t)
        dv = np.asarray(command.dv[0], dtype=np.float64)
        assert np.linalg.norm(dv) <= CAP * (1.0 + 1e-5)
        own = a @ own + b @ dv
        opponent = a @ opponent
    assert np.linalg.norm(own[:3] - opponent[:3]) < 5.0


def test_policies_are_hashable_and_registered():
    command_cls = make_impulsive_maneuver_command_cls(1)
    lqr = LQRFeedback.build(
        mean_motion=MEAN_MOTION,
        dt=DT,
        n_vehicles=1,
        n_opponents=1,
        command_cls=command_cls,
        max_dv_mps=CAP,
    )
    assert isinstance(hash(lqr), int)
    assert resolve(PolicyKey.PD_FEEDBACK) is PDFeedback
    assert resolve(PolicyKey.LQR_FEEDBACK) is LQRFeedback
