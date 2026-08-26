"""J2-only ECI orbit dynamics.

Hand-rolled J2 acceleration so we don't depend on astrojax shipping a
J2-coefficient ``GravityModel``. The full N×N spherical-harmonics path is
exposed via ``ASTROJAX_ORBIT`` (Phase 7).

Acceleration in ECI:
  a_kepler = -mu * r / |r|^3
  a_j2 = -3/2 * J2 * mu * R_E^2 / |r|^5 * (
      (1 - 5 z^2/|r|^2) * x,
      (1 - 5 z^2/|r|^2) * y,
      (3 - 5 z^2/|r|^2) * z,
  )
"""

from __future__ import annotations

import astrojax
import jax
import jax.numpy as jnp

from orbitalgym.registry import DynamicsKey, DynamicsKind, Frame, register

_MU = 3.986004418e14
_J2 = float(astrojax.J2_EARTH)
_RE = float(astrojax.R_EARTH)


def _rhs(t, state):  # signature matches astrojax integrator contract
    r = state[:3]
    v = state[3:]
    r_norm = jnp.linalg.norm(r)
    a_kep = -_MU * r / r_norm**3
    z2_over_r2 = (r[2] / r_norm) ** 2
    factor = -1.5 * _J2 * _MU * _RE**2 / r_norm**5
    a_j2 = factor * jnp.array(
        [
            (1.0 - 5.0 * z2_over_r2) * r[0],
            (1.0 - 5.0 * z2_over_r2) * r[1],
            (3.0 - 5.0 * z2_over_r2) * r[2],
        ]
    )
    return jnp.concatenate([v, a_kep + a_j2])


def _step_one(state6: jax.Array, dt: float) -> jax.Array:
    """Single RK4 step on a 6D ECI state vector under Keplerian + J2.

    ``rk4_step`` returns a ``StepResult`` namedtuple — we keep only ``state``.
    """
    return astrojax.rk4_step(_rhs, 0.0, state6, dt).state


@register(DynamicsKey.J2_ECI, frame=Frame.ECI, kind=DynamicsKind.ABSOLUTE)
def j2_eci_step(state: jax.Array, dv: jax.Array, params, dt: float) -> jax.Array:
    """J2-only ECI step.

    state: (n, 6) — (rx, ry, rz, vx, vy, vz) per vehicle in ECI [m, m/s].
    dv:    (n, 3) — (dvx, dvy, dvz) impulsive Δv applied at start of interval.
    params: unused (kept for signature parity with other dynamics).
    dt:    scalar seconds.
    """
    s0 = state.at[:, 3:].add(dv)
    return jax.vmap(_step_one, in_axes=(0, None))(s0, dt)
