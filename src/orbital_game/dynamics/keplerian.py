"""Keplerian (point-mass two-body) orbit dynamics in ECI.

Wraps ``astrojax.create_orbit_dynamics`` with the default ``ForceModelConfig``
(point-mass two-body) and integrates one env step with ``rk4_step``.

Δv is applied at the START of the interval (impulsive convention shared with
HCW), so impulsive and continuous-actuator step functions remain
interchangeable under a common signature.
"""

from __future__ import annotations

import astrojax
import jax
from astrojax import Epoch
from astrojax.eop import zero_eop

from orbital_game.registry import DynamicsKey, DynamicsKind, Frame, register

# Build the ECI RHS once at module load. The reference epoch is irrelevant for
# point-mass two-body (the dynamics has no explicit time dependence), so we
# anchor at J2000.
_EPOCH_REF = Epoch(2000, 1, 1, 12, 0, 0.0)
_RHS = astrojax.create_orbit_dynamics(eop=zero_eop(), epoch_0=_EPOCH_REF)


def _step_one(state6: jax.Array, dt: float) -> jax.Array:
    """Single RK4 step on a 6D ECI state vector.

    ``t`` is irrelevant for point-mass two-body; we pass 0.0. ``rk4_step``
    returns a ``StepResult`` namedtuple — we keep only ``state``.
    """
    return astrojax.rk4_step(_RHS, 0.0, state6, dt).state


@register(DynamicsKey.KEPLERIAN_ECI, frame=Frame.ECI, kind=DynamicsKind.ABSOLUTE)
def keplerian_eci_step(state: jax.Array, dv: jax.Array, params, dt: float) -> jax.Array:
    """Keplerian (point-mass) ECI step.

    state: (n, 6) — (rx, ry, rz, vx, vy, vz) per vehicle in ECI [m, m/s].
    dv:    (n, 3) — (dvx, dvy, dvz) impulsive Δv applied at start of interval.
    params: unused (kept for signature parity with other dynamics).
    dt:    scalar seconds.
    """
    s0 = state.at[:, 3:].add(dv)
    return jax.vmap(_step_one, in_axes=(0, None))(s0, dt)
