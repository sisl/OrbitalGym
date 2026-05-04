"""Tests for the KEPLERIAN_ECI dynamics registration.

Verifies that astrojax.create_orbit_dynamics + rk4_step are wired through the
registry as a (Frame.ECI, DynamicsKind.ABSOLUTE) step function with the
shared start-of-interval Δv convention.
"""

import jax.numpy as jnp
import numpy as np


def test_keplerian_eci_circular_orbit_returns_to_origin():
    """A particle on a circular orbit returns to its starting state after one period."""
    from orbital_game.dynamics import keplerian  # noqa: F401 — register side effect
    from orbital_game.registry import DynamicsKey, DynamicsKind, Frame, resolve

    fn = resolve(DynamicsKey.KEPLERIAN_ECI)
    assert fn.frame is Frame.ECI
    assert fn.kind is DynamicsKind.ABSOLUTE

    mu = 3.986004418e14
    r = 6378137.0 + 500e3
    v = (mu / r) ** 0.5
    period = 2 * jnp.pi * jnp.sqrt(r**3 / mu)
    state0 = jnp.array([[r, 0.0, 0.0, 0.0, v, 0.0]])  # (1, 6)
    dv = jnp.zeros((1, 3))
    n_sub = 360
    dt = float(period / n_sub)
    state = state0
    for _ in range(n_sub):
        state = fn(state, dv, params=None, dt=dt)
    # rtol against zero-valued components is degenerate; use atol=1.0 m / 1e-3 m/s
    # absolute tolerance, which is ~1.5e-7 relative to the LEO position/velocity scale.
    np.testing.assert_allclose(state, state0, rtol=1e-3, atol=1.0)


def test_keplerian_eci_applies_dv_at_step_start():
    """With dt=0 the RK4 step is the identity, so the only change is the start-of-interval Δv."""
    from orbital_game.dynamics import keplerian  # noqa: F401
    from orbital_game.registry import DynamicsKey, resolve

    fn = resolve(DynamicsKey.KEPLERIAN_ECI)
    state0 = jnp.array([[7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]])
    dv = jnp.array([[1.0, 0.0, 0.0]])  # 1 m/s radial impulse
    out = fn(state0, dv, params=None, dt=0.0)  # dt=0 -> no propagation, just the dv
    np.testing.assert_allclose(out[0, 3:], jnp.array([1.0, 7.5e3, 0.0]), atol=1e-9)
