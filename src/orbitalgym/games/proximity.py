"""Within-step closest-approach proximity events.

Catch and breach events in the Lady-Bandit-Guard game are geometric facts
about the *continuous* trajectory, not about the sampled endpoints of a
decision step. A guard and a bandit can pass within metres of each other
between two 10 s samples and both samples land far outside the catch
radius.

Within one step the relative motion is modelled as a straight line from the
pre-step relative position ``r0`` with the chord velocity
``v = (r1 - r0) / dt``. HCW curvature over a 10 s step on a 7000 km orbit
displaces the true path from that chord by well under 0.1 m, so the chord
is an accurate stand-in for the arc at the scale of the event radii.

An event additionally requires the relative speed to be below a threshold:
a high-speed flyby through the sphere is a near miss, not a capture or a
breach. Setting the speed threshold to infinity recovers pure radius
gating.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_EPS = 1e-12


def positions(side_state) -> jax.Array:
    """Per-vehicle ``(..., N, 3)`` positions, zero-padded from a 2D ``rt`` layout.

    Leading axes pass through, so a time-stacked side state returns
    ``(T, N, 3)``.
    """
    if hasattr(side_state, "rtn"):
        return side_state.rtn[..., :3]
    rt = side_state.rt[..., :2]
    return jnp.concatenate([rt, jnp.zeros_like(rt[..., :1])], axis=-1)


def closest_approach(r0: jax.Array, v: jax.Array, dt: float) -> tuple[jax.Array, jax.Array]:
    """Minimum distance over a coast of length ``dt``, and the relative speed.

    Args:
        r0: Relative position at the start of the step, shape ``(..., 3)``.
        v: Constant relative velocity over the step, shape ``(..., 3)``.
        dt: Step duration in seconds.

    Returns:
        ``(distance, speed)``, both shape ``(...)``: the minimum of
        ``|r0 + v * s|`` over ``s`` in ``[0, dt]``, and ``|v|``.
    """
    r0 = jnp.asarray(r0)
    v = jnp.asarray(v)
    vv = jnp.sum(v * v, axis=-1)
    t_star = jnp.clip(-jnp.sum(r0 * v, axis=-1) / (vv + _EPS), 0.0, dt)
    closest = r0 + v * t_star[..., None]
    return jnp.linalg.norm(closest, axis=-1), jnp.sqrt(vv)


def proximity_event(
    r0: jax.Array,
    v: jax.Array,
    dt: float,
    radius_m: float,
    speed_mps: float,
) -> jax.Array:
    """Whether the coast passes within ``radius_m`` while slower than ``speed_mps``."""
    distance, speed = closest_approach(r0, v, dt)
    return jnp.logical_and(distance < radius_m, speed < speed_mps)


def lbg_events_from_positions(
    guard_prev: jax.Array,
    guard_next: jax.Array,
    bandit_prev: jax.Array,
    bandit_next: jax.Array,
    dt: float,
    catch_radius_m: float,
    catch_speed_mps: float,
    breach_radius_m: float,
    breach_speed_mps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Catch/breach events and closest-approach distances from position arrays.

    Positions are ``(..., n_g, 3)`` and ``(..., n_b, 3)``; leading axes pass
    through, so a whole trajectory is scored in one call. An event holds when
    *some* pair is both inside the radius and slower than the threshold — the
    per-pair conjunction, not the closest pair's speed.

    Returns:
        ``(caught, breached, d_gb_min, d_bl_min)`` with the leading shape of
        the inputs.
    """
    rel_gb = guard_prev[..., :, None, :] - bandit_prev[..., None, :, :]
    rel_gb_next = guard_next[..., :, None, :] - bandit_next[..., None, :, :]
    d_gb, speed_gb = closest_approach(rel_gb, (rel_gb_next - rel_gb) / dt, dt)
    d_gb_min = jnp.min(d_gb, axis=(-2, -1))

    d_bl, speed_bl = closest_approach(bandit_prev, (bandit_next - bandit_prev) / dt, dt)
    d_bl_min = jnp.min(d_bl, axis=-1)

    if catch_radius_m > 0.0:
        caught = jnp.any(
            jnp.logical_and(d_gb < catch_radius_m, speed_gb < catch_speed_mps), axis=(-2, -1)
        )
    else:
        caught = jnp.zeros(d_gb_min.shape, dtype=bool)
    breached = jnp.any(
        jnp.logical_and(d_bl < breach_radius_m, speed_bl < breach_speed_mps), axis=-1
    )

    return caught, breached, d_gb_min, d_bl_min


def lbg_events(
    prev_state,
    next_state,
    dt: float,
    catch_radius_m: float,
    catch_speed_mps: float,
    breach_radius_m: float,
    breach_speed_mps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Catch/breach events and closest-approach distances over one step.

    Catch geometry is guard minus bandit over every guard-bandit pair;
    breach geometry is bandit minus lady, the RTN origin. ``catch_radius_m``
    of zero disables the catch event while still reporting ``d_gb_min``.

    Returns:
        ``(caught, breached, d_gb_min, d_bl_min)`` — two scalar booleans and
        the two minimum closest-approach distances over the step.
    """
    return lbg_events_from_positions(
        positions(prev_state.guards),
        positions(next_state.guards),
        positions(prev_state.bandits),
        positions(next_state.bandits),
        dt,
        catch_radius_m,
        catch_speed_mps,
        breach_radius_m,
        breach_speed_mps,
    )
