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

An event may further require *dwell*: a bandit that holds the radius for a
single step has only brushed it, while one that holds it for several
consecutive steps has been kept there. :func:`lbg_dwell_step` advances the
per-bandit consecutive-step counters the environment state carries and
:func:`lbg_events_with_dwell` turns them into the events, with a dwell of
zero recovering the single-step condition exactly.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.reference_orbit import mean_motion_host

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


def lbg_proximity_masks(
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
    """Per-bandit in-radius masks and closest-approach distances.

    Positions are ``(..., n_g, 3)`` and ``(..., n_b, 3)``; leading axes pass
    through, so a whole trajectory is scored in one call. A bandit is inside
    the catch radius when *some* guard both passes within ``catch_radius_m``
    of it and is slower than ``catch_speed_mps`` over the step — the per-pair
    conjunction, not the closest pair's speed. It is inside the breach radius
    when its own pass at the lady meets the same two conditions.

    Returns:
        ``(in_catch, in_breach, d_gb_min, d_bl_min)``. The two masks are
        ``(..., n_b)``; the two distances are the minima over every pair and
        over every bandit, with the leading shape of the inputs.
    """
    rel_gb = guard_prev[..., :, None, :] - bandit_prev[..., None, :, :]
    rel_gb_next = guard_next[..., :, None, :] - bandit_next[..., None, :, :]
    d_gb, speed_gb = closest_approach(rel_gb, (rel_gb_next - rel_gb) / dt, dt)
    d_gb_min = jnp.min(d_gb, axis=(-2, -1))

    d_bl, speed_bl = closest_approach(bandit_prev, (bandit_next - bandit_prev) / dt, dt)
    d_bl_min = jnp.min(d_bl, axis=-1)

    if catch_radius_m > 0.0:
        in_catch = jnp.any(
            jnp.logical_and(d_gb < catch_radius_m, speed_gb < catch_speed_mps), axis=-2
        )
    else:
        in_catch = jnp.zeros(d_bl.shape, dtype=bool)
    in_breach = jnp.logical_and(d_bl < breach_radius_m, speed_bl < breach_speed_mps)

    return in_catch, in_breach, d_gb_min, d_bl_min


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

    The events are :func:`lbg_proximity_masks` reduced over the bandits: an
    event holds when some bandit is inside the corresponding radius.

    Returns:
        ``(caught, breached, d_gb_min, d_bl_min)`` with the leading shape of
        the inputs.
    """
    in_catch, in_breach, d_gb_min, d_bl_min = lbg_proximity_masks(
        guard_prev,
        guard_next,
        bandit_prev,
        bandit_next,
        dt,
        catch_radius_m,
        catch_speed_mps,
        breach_radius_m,
        breach_speed_mps,
    )
    return jnp.any(in_catch, axis=-1), jnp.any(in_breach, axis=-1), d_gb_min, d_bl_min


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


def advance_dwell(count: jax.Array, inside: jax.Array) -> jax.Array:
    """One step of a consecutive-step counter: increment inside, reset outside.

    ``count`` and ``inside`` are per-bandit arrays of the same shape. The
    result counts how many consecutive steps, up to and including this one,
    each bandit has spent inside the radius; a single step outside sends it
    back to zero.

    A step counts as inside when its *within-step closest approach* is
    inside, the same test the single-step event uses. The step a bandit
    departs on therefore still counts — it was inside for part of it — and
    the first step spent wholly outside is the one that breaks the hold.
    """
    return jnp.where(inside, count + 1, jnp.zeros_like(count))


def lbg_dwell_step(
    prev_state,
    next_state,
    dt: float,
    catch_radius_m: float,
    catch_speed_mps: float,
    breach_radius_m: float,
    breach_speed_mps: float,
) -> tuple[jax.Array, jax.Array]:
    """The dwell counters leaving a step, from the counters entering it.

    Advances ``prev_state.dwell_catch`` and ``prev_state.dwell_breach`` with
    :func:`advance_dwell` on the per-bandit masks
    :func:`lbg_proximity_masks` reports for the step. A step whose closest
    approach fails either the radius or the speed gate resets that counter,
    so a positive count means the gate held on every one of those steps.

    Returns:
        ``(dwell_catch, dwell_breach)``, both ``(..., n_b)`` integers.
    """
    in_catch, in_breach, _, _ = lbg_proximity_masks(
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
    return (
        advance_dwell(prev_state.dwell_catch, in_catch),
        advance_dwell(prev_state.dwell_breach, in_breach),
    )


def lbg_events_with_dwell(
    prev_state,
    next_state,
    dt: float,
    catch_radius_m: float,
    catch_speed_mps: float,
    breach_radius_m: float,
    breach_speed_mps: float,
    catch_dwell_steps: int,
    breach_dwell_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Catch/breach events under an optional consecutive-step dwell requirement.

    A dwell of zero leaves the event exactly as :func:`lbg_events` reports
    it: the within-step closest approach on this step alone. A positive
    dwell replaces it with the requirement that some bandit has held the
    radius — and the speed gate, which enters through the counter — for that
    many consecutive steps, read off the counters ``next_state`` carries.

    Returns:
        ``(caught, breached, d_gb_min, d_bl_min)``; the distances are the
        step's own closest approaches either way.
    """
    caught, breached, d_gb_min, d_bl_min = lbg_events(
        prev_state,
        next_state,
        dt,
        catch_radius_m,
        catch_speed_mps,
        breach_radius_m,
        breach_speed_mps,
    )
    if catch_dwell_steps > 0:
        caught = jnp.any(next_state.dwell_catch >= catch_dwell_steps, axis=-1)
    if breach_dwell_steps > 0:
        breached = jnp.any(next_state.dwell_breach >= breach_dwell_steps, axis=-1)
    return caught, breached, d_gb_min, d_bl_min


def stm_power_stack(mean_motion: float, dt: float, max_steps: int) -> np.ndarray:
    """Powers ``phi^0 ... phi^max_steps`` of the one-step HCW-RTN STM, ``(max_steps + 1, 6, 6)``.

    Coasting a state to every step of the horizon is one batched matmul
    against this stack rather than a chain of ``max_steps`` dependent ones,
    so :func:`ballistic_breach_possible` costs the same whichever step of the
    episode calls it. The stack is built with an associative scan under
    compile-time evaluation and returned as a concrete NumPy array, so it is
    memoised on its three scalar arguments (fixed for a scenario) and is safe
    to build from inside a ``jax.jit`` trace without leaking a tracer.

    Args:
        mean_motion: Reference-orbit mean motion in rad/s.
        dt: Step duration in seconds.
        max_steps: Highest power to build.
    """
    with jax.ensure_compile_time_eval():
        phi = hcw_rtn_stm(mean_motion, dt)
        powers = jax.lax.associative_scan(jnp.matmul, jnp.broadcast_to(phi, (max_steps, 6, 6)))
        stack = jnp.concatenate([jnp.eye(6, dtype=phi.dtype)[None], powers])
    return np.asarray(stack)


_stm_power_stack_cached = lru_cache(maxsize=8)(stm_power_stack)


def ballistic_breach_possible(
    x_rtn: jax.Array,
    phi_powers: jax.Array | np.ndarray,
    n_remaining: jax.Array,
    dt: float,
    radius_m: float,
    speed_mps: float,
) -> jax.Array:
    """Whether a coasting RTN state can still reach the lady before the clock runs out.

    Coasts ``x_rtn`` to every step of the horizon at once through
    ``phi_powers``, then tests :func:`proximity_event` against the origin on
    each of the resulting segments, keeping only the first ``n_remaining`` of
    them. The horizon length is static — it is the length of ``phi_powers`` —
    while ``n_remaining`` is traced, so one compiled evaluation serves every
    step of an episode.

    Args:
        x_rtn: Coasting state ``(..., 6)`` as ``[R, T, N, Rdot, Tdot, Ndot]``.
        phi_powers: Powers of the one-step STM, ``(max_steps + 1, 6, 6)``, from
            :func:`stm_power_stack`.
        n_remaining: Steps left in the episode; segments beyond it are masked.
        dt: Step duration in seconds, matching ``phi_powers``.
        radius_m: Breach radius about the origin.
        speed_mps: Relative-speed gate for the breach.

    Returns:
        Boolean with the leading shape of ``x_rtn``.
    """
    dtype = jnp.result_type(x_rtn, phi_powers)
    x_rtn = jnp.asarray(x_rtn, dtype=dtype)
    phi_powers = jnp.asarray(phi_powers, dtype=dtype)

    coast = jnp.einsum("kij,...j->...ki", phi_powers, x_rtn)  # (..., max_steps + 1, 6)
    r0 = coast[..., :-1, :3]
    v = (coast[..., 1:, :3] - r0) / dt
    unmasked = proximity_event(r0, v, dt, radius_m, speed_mps)
    return jnp.any(jnp.logical_and(unmasked, jnp.arange(r0.shape[-2]) < n_remaining), axis=-1)


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
        powers = _stm_power_stack_cached(
            mean_motion_host(params.reference_orbit), float(params.dt), int(params.max_steps)
        )
        reachable = ballistic_breach_possible(
            bandits.rtn,
            powers,
            params.max_steps - state.step,
            params.dt,
            breach_radius_m,
            breach_speed_mps,
        )
        stranded = jnp.logical_and(bandits.propellant_mass <= 0.0, jnp.logical_not(reachable))
        repelled = jnp.logical_or(repelled, stranded)
    return jnp.all(repelled, axis=-1)
