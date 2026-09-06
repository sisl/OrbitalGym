"""HCW 3D (RTN) dynamics: cross-track is a decoupled sinusoid of period 2π/n."""

import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym.dynamics.hcw import hcw_rtn_step, hcw_rtn_stm


def test_cross_track_is_sinusoidal_and_decoupled():
    mu = 3.986004418e14
    a = 7000e3
    n = jnp.sqrt(mu / a**3)

    # Pure cross-track IC: N0 = 100 m, Ndot0 = 0. All in-plane components zero.
    state0 = jnp.array([[0.0, 0.0, 100.0, 0.0, 0.0, 0.0]])

    class _Params:
        mean_motion = n

    # Step through a full period in 360 substeps.
    period = 2 * jnp.pi / n
    dt = period / 360

    zero_impulse = jnp.zeros((1, 3))
    state = state0
    z_samples = [float(state[0, 2])]
    for _ in range(360):
        state = hcw_rtn_step(state, zero_impulse, _Params(), float(dt))
        z_samples.append(float(state[0, 2]))

    z_max = max(abs(z) for z in z_samples)
    z_min = min(z_samples)
    # Pure cross-track motion: amplitude conserved, traverses -100..+100 range.
    assert abs(z_max - 100.0) < 1.0
    assert z_min < -99.0
    # And in-plane components must remain zero (no coupling).
    assert abs(float(state[0, 0])) < 1e-6
    assert abs(float(state[0, 1])) < 1e-6


def test_rtn_impulse_adds_to_velocity_components():
    class _Params:
        mean_motion = 0.001

    state0 = jnp.zeros((1, 6))
    impulse = jnp.array([[1.0, 2.0, 3.0]])
    new = hcw_rtn_step(state0, impulse, _Params(), dt=1e-6)
    assert float(new[0, 3]) == pytest.approx(1.0, abs=1e-3)
    assert float(new[0, 4]) == pytest.approx(2.0, abs=1e-3)
    assert float(new[0, 5]) == pytest.approx(3.0, abs=1e-3)


def test_hcw_rtn_stm_matches_step_with_zero_impulse():
    """The closed-form 6x6 STM and the per-step propagator must agree.

    With zero impulse, hcw_rtn_step(x) and hcw_rtn_stm @ x should produce the
    same next state to numerical precision. This pins the STM helper (used by
    Kalman filters and controller builds) to the dynamics actually used inside the
    env, preventing the two from drifting apart silently.
    """
    n = 1.0e-3
    dt = 30.0

    class _Params:
        mean_motion = n

    rng = np.random.default_rng(0)
    states = jnp.asarray(rng.standard_normal((4, 6)) * jnp.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0]))
    zero_impulse = jnp.zeros((4, 3))

    stepped = hcw_rtn_step(states, zero_impulse, _Params(), dt)
    phi = hcw_rtn_stm(n, dt)
    matmul = states @ phi.T

    np.testing.assert_allclose(np.asarray(stepped), np.asarray(matmul), rtol=1e-10, atol=1e-10)


def test_hcw_rtn_stm_block_structure():
    """The 6x6 STM should be block-diagonal between in-plane (R,T,Rdot,Tdot)
    and cross-track (N,Ndot) — coupling those would imply non-physical
    transfer between modes the HCW model treats as independent.
    """
    phi = np.asarray(hcw_rtn_stm(1.0e-3, 30.0))
    in_idx = np.array([0, 1, 3, 4])
    out_idx = np.array([2, 5])
    # Off-block entries (in-plane <-> cross-track) must be exactly zero.
    assert np.all(phi[np.ix_(in_idx, out_idx)] == 0.0)
    assert np.all(phi[np.ix_(out_idx, in_idx)] == 0.0)
