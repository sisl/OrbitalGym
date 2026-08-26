"""Frame-conversion helper shared by SB and OB game rewards.

Composes astrojax primitives to convert a vehicle's RTN-relative state
to its ECI position. The reference orbit is propagated via two-body
Keplerian motion from epoch.

Internal to games/ — the spec rule (Q8) is that games invoke astrojax
directly. If a similar helper is needed outside games/, prefer adding
the primitive to astrojax rather than promoting this module.
"""

from __future__ import annotations

import astrojax
import jax.numpy as jnp


def reference_orbit_eci_at_t(reference_orbit, epoch_mjd_utc: float, t_offset_s):
    """Propagate the reference orbit (initial ECI state at epoch) by t_offset_s
    using two-body Keplerian motion. Returns (ECI position (3,), velocity (3,))."""
    state0 = jnp.concatenate([reference_orbit.position_eci, reference_orbit.velocity_eci])
    koe = astrojax.state_eci_to_koe(state0)
    # Advance mean anomaly by mean_motion * t_offset_s.
    n = jnp.sqrt(astrojax.GM_EARTH / koe[0] ** 3)
    new_mean_anomaly = koe[5] + n * t_offset_s
    new_koe = koe.at[5].set(new_mean_anomaly)
    new_state = astrojax.state_koe_to_eci(new_koe)
    return new_state[:3], new_state[3:]


def rtn_basis(reference_pos_eci, reference_vel_eci):
    """Compute the RTN basis vectors at a reference-orbit ECI state.

        R̂ = position / |position|
        N̂ = (position × velocity) / |position × velocity|
        T̂ = N̂ × R̂

    Returns a (3, 3) rotation matrix whose columns are R̂, T̂, N̂ — i.e.,
    `R_rtn_to_eci @ rtn_vec = eci_vec`.
    """
    ref_eci_state = jnp.concatenate([reference_pos_eci, reference_vel_eci])
    return astrojax.rotation_rtn_to_eci(ref_eci_state)


def rtn_offset_to_eci(rtn_offset, reference_pos_eci, reference_vel_eci):
    """Convert an RTN-frame relative position (3,) to ECI position (3,).

    eci = reference_pos_eci + R_rtn_to_eci @ rtn_offset
    """
    rot = rtn_basis(reference_pos_eci, reference_vel_eci)
    return reference_pos_eci + rot @ rtn_offset


def vehicle_eci_position(vehicle_state, reference_orbit, epoch_mjd_utc, t_offset_s):
    """Convert a single vehicle's RTN state to its ECI position.

    Assumes vehicle_state.rtn is shape (..., 6) where the first 3 are
    radial/tangential/normal position offsets in meters. Falls back to
    `vehicle_state.rt` (4-dim, 2D RT) if .rtn is not present, padding the
    normal axis with 0.
    """
    if hasattr(vehicle_state, "rtn"):
        rtn = vehicle_state.rtn
        offset = rtn[:3]
    else:
        rt = vehicle_state.rt
        offset = jnp.array([rt[0], rt[1], 0.0])
    ref_pos, ref_vel = reference_orbit_eci_at_t(reference_orbit, epoch_mjd_utc, t_offset_s)
    return rtn_offset_to_eci(offset, ref_pos, ref_vel)
