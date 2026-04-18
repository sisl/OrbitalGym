"""HCW 2D (RT) dynamics — classic closed-orbit sanity test.

For a chief orbit with mean motion n, the relative motion in the RT (in-plane)
sub-frame is bounded exactly when:
    y_dot_0 = -2 n x_0    (along-track initial velocity condition)
    x_dot_0 = (n / 2) y_0 (radial initial velocity condition)

With these, the drift cancels and the motion is a closed ellipse with period 2π/n.
We test that without control, the state over 10 orbital periods has bounded magnitude.
"""

import jax.numpy as jnp
import pytest

from orbital_game.dynamics.hcw import hcw_rt_step


def test_hcw_rt_closed_orbit_remains_bounded():
    # Chief at 7000 km circular; n = sqrt(mu / a^3).
    mu = 3.986004418e14
    a = 7000e3
    n = jnp.sqrt(mu / a**3)
    period = 2 * jnp.pi / n

    # Closed-orbit IC: x0 = 1 km radial, y0 = 0, xdot0 = 0, ydot0 = -2 n x0
    x0 = 1000.0
    y0 = 0.0
    xdot0 = 0.0
    ydot0 = -2.0 * n * x0
    state0 = jnp.array([[x0, y0, xdot0, ydot0]])  # (1, 4)

    # Simulate 10 periods with dt = period / 100.
    dt = period / 100
    n_steps = int(10 * 100)

    class _Params:
        mean_motion = n

    state = state0
    zero_impulse = jnp.zeros((1, 2))
    for _ in range(n_steps):
        state = hcw_rt_step(state, zero_impulse, _Params(), float(dt))

    # Closed orbit magnitude: max |x| ~= x0, max |y| ~= 2 x0. Should stay under 3 km.
    mag = jnp.max(jnp.linalg.norm(state[0, :2]))
    assert mag < 3000.0


def test_hcw_rt_applies_impulse_to_velocity_components():
    """An impulse of (dvx, dvy) should add directly to (xdot, ydot) at the start of the step."""
    n = 0.001  # rad/s (arbitrary)

    class _Params:
        mean_motion = n

    state0 = jnp.zeros((1, 4))
    impulse = jnp.array([[2.0, 3.0]])  # dvx=2, dvy=3
    new = hcw_rt_step(state0, impulse, _Params(), dt=1e-6)
    # Over a ~0 dt the impulse should be almost entirely reflected in new velocities.
    assert float(new[0, 2]) == pytest.approx(2.0, abs=1e-3)
    assert float(new[0, 3]) == pytest.approx(3.0, abs=1e-3)
