"""Frame conversions, vmapped over vehicles.

`eci` arrays are shape (n, 6); `rtn` arrays are shape (n, 6); `ref_eci` is shape (6,).
The reference state is the chief satellite — astrojax's RTN frame is anchored to it.
"""

from __future__ import annotations

import astrojax
import jax
import jax.numpy as jnp

from orbital_game.registry import Frame


def eci_to_rtn(eci: jax.Array, ref_eci: jax.Array) -> jax.Array:
    """eci: (n, 6) deputy states; ref_eci: (6,) chief state. Returns (n, 6) RTN."""
    return jax.vmap(astrojax.state_eci_to_rtn, in_axes=(None, 0))(ref_eci, eci)


def rtn_to_eci(rtn: jax.Array, ref_eci: jax.Array) -> jax.Array:
    """Inverse of eci_to_rtn."""
    return jax.vmap(astrojax.state_rtn_to_eci, in_axes=(None, 0))(ref_eci, rtn)


def convert_state(state: jax.Array, src: Frame, dst: Frame, ref_eci: jax.Array) -> jax.Array:
    """Dispatch by (src, dst). Identity when src == dst.

    RT <-> RTN handled by zero-pad / drop of cross-track. RT/RTN <-> ECI goes
    through RTN as the canonical relative frame.
    """
    if src is dst:
        return state
    if src is Frame.ECI and dst is Frame.RTN:
        return eci_to_rtn(state, ref_eci)
    if src is Frame.RTN and dst is Frame.ECI:
        return rtn_to_eci(state, ref_eci)
    if src is Frame.RT and dst is Frame.RTN:
        # (n, 4) [r, t, rdot, tdot] -> (n, 6) [r, t, 0, rdot, tdot, 0]
        n = state.shape[0]
        out = jnp.zeros((n, 6))
        out = out.at[:, 0].set(state[:, 0])  # r
        out = out.at[:, 1].set(state[:, 1])  # t
        out = out.at[:, 3].set(state[:, 2])  # rdot
        out = out.at[:, 4].set(state[:, 3])  # tdot
        return out
    if src is Frame.RTN and dst is Frame.RT:
        return state[:, [0, 1, 3, 4]]
    if src is Frame.RT and dst is Frame.ECI:
        return rtn_to_eci(convert_state(state, Frame.RT, Frame.RTN, ref_eci), ref_eci)
    if src is Frame.ECI and dst is Frame.RT:
        return convert_state(eci_to_rtn(state, ref_eci), Frame.RTN, Frame.RT, ref_eci)
    raise ValueError(f"Unsupported frame conversion {src.value} -> {dst.value}")


def convert_action(dv: jax.Array, src: Frame, dst: Frame, ref_eci: jax.Array) -> jax.Array:
    """Per-vehicle Δv rotation.

    dv: (n, k). For RT, k=2; for RTN/ECI, k=3. RTN <-> ECI uses the rotation
    matrix from astrojax. RT <-> RTN pads/drops the third (cross-track) component.
    """
    if src is dst:
        return dv
    if src is Frame.RTN and dst is Frame.ECI:
        R = astrojax.rotation_rtn_to_eci(ref_eci)  # noqa: N806 — rotation matrix; uppercase by convention
        return dv @ R.T
    if src is Frame.ECI and dst is Frame.RTN:
        R = astrojax.rotation_eci_to_rtn(ref_eci)  # noqa: N806 — rotation matrix
        return dv @ R.T
    if src is Frame.RT and dst is Frame.RTN:
        n = dv.shape[0]
        out = jnp.zeros((n, 3))
        return out.at[:, :2].set(dv)
    if src is Frame.RTN and dst is Frame.RT:
        return dv[:, :2]
    # RT ↔ ECI is intentionally lossy: RT → RTN pads cross-track with zero,
    # RTN → RT drops it. So RT → ECI → RT zeroes the cross-track of any
    # RTN-padded intermediate. This is by design: an RT-frame action has no
    # cross-track component, and round-tripping through ECI and back must
    # preserve that.
    if src is Frame.RT and dst is Frame.ECI:
        return convert_action(
            convert_action(dv, Frame.RT, Frame.RTN, ref_eci), Frame.RTN, Frame.ECI, ref_eci
        )
    if src is Frame.ECI and dst is Frame.RT:
        return convert_action(
            convert_action(dv, Frame.ECI, Frame.RTN, ref_eci), Frame.RTN, Frame.RT, ref_eci
        )
    raise ValueError(f"Unsupported action conversion {src.value} -> {dst.value}")


def rtn_initial_conditions_to_eci(rtn_state: jax.Array, ref_eci: jax.Array) -> jax.Array:
    """Used by callers that sample ICs in RTN but need ECI truth state."""
    return rtn_to_eci(rtn_state, ref_eci)
