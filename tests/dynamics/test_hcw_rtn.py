"""HCW 3D (RTN) dynamics: cross-track is a decoupled sinusoid of period 2π/n."""

import jax.numpy as jnp
import pytest

from orbital_game.dynamics.hcw import hcw_rtn_step


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
