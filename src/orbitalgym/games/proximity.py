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

from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.reference_orbit import mean_motion

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


def ballistic_breach_possible(
    x_rtn: jax.Array,
    phi: jax.Array,
    n_remaining: jax.Array,
    max_steps: int,
    dt: float,
    radius_m: float,
    speed_mps: float,
) -> jax.Array:
    """Whether a coasting RTN state can still reach the lady before the clock runs out.

    Propagates ``x_rtn`` under the one-step state transition matrix ``phi``
    for ``max_steps`` steps and tests :func:`proximity_event` against the
    origin on each segment, keeping only the first ``n_remaining`` of them.
    ``max_steps`` is the static scan length; ``n_remaining`` is traced, so
    the same compiled scan serves every step of an episode.

    Args:
        x_rtn: Coasting state ``(..., 6)`` as ``[R, T, N, Rdot, Tdot, Ndot]``.
        phi: One-step ``(6, 6)`` state transition matrix.
        n_remaining: Steps left in the episode; segments beyond it are masked.
        max_steps: Static upper bound on the number of segments scanned.
        dt: Step duration in seconds, matching ``phi``.
        radius_m: Breach radius about the origin.
        speed_mps: Relative-speed gate for the breach.

    Returns:
        Boolean with the leading shape of ``x_rtn``.
    """

    dtype = jnp.result_type(x_rtn, phi)
    x_rtn = jnp.asarray(x_rtn, dtype=dtype)
    phi = jnp.asarray(phi, dtype=dtype)

    def step(x, k):
        x_next = x @ phi.T
        r0 = x[..., :3]
        v = (x_next[..., :3] - r0) / dt
        hit = jnp.logical_and(proximity_event(r0, v, dt, radius_m, speed_mps), k < n_remaining)
        return x_next, hit

    _, hits = jax.lax.scan(step, x_rtn, jnp.arange(max_steps))
    return jnp.any(hits, axis=0)


def lbg_repelled(
    state,
    params,
    escape_radius_m: float,
    repel_on_empty_tank: bool,
    breach_radius_m: float,
    breach_speed_mps: float,
) -> jax.Array:
    """Whether every bandit has been repelled — no longer a threat to the lady.

    A bandit is repelled when it is farther than ``escape_radius_m`` from the
    lady (zero disables that gate), or when ``repel_on_empty_tank`` is set and
    it has burnt its propellant on a coast that cannot reach the lady within
    the episode's remaining steps. Both gates off returns ``False``.

    ``params`` is the scenario config: ``dt``, ``max_steps`` and, for the
    empty-tank gate, ``reference_orbit`` supply the coast model.
    """
    if escape_radius_m <= 0.0 and not repel_on_empty_tank:
        return jnp.asarray(False)

    bandits = state.bandits
    r_bandits = positions(bandits)
    repelled = jnp.zeros(r_bandits.shape[:-1], dtype=bool)
    if escape_radius_m > 0.0:
        repelled = jnp.linalg.norm(r_bandits, axis=-1) > escape_radius_m
    if repel_on_empty_tank:
        phi = hcw_rtn_stm(float(mean_motion(params.reference_orbit)), params.dt)
        reachable = ballistic_breach_possible(
            bandits.rtn,
            phi,
            params.max_steps - state.step,
            params.max_steps,
            params.dt,
            breach_radius_m,
            breach_speed_mps,
        )
        stranded = jnp.logical_and(bandits.propellant_mass <= 0.0, jnp.logical_not(reachable))
        repelled = jnp.logical_or(repelled, stranded)
    return jnp.all(repelled, axis=-1)
