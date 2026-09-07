"""Per-episode outcome classification and resource accounting for LBG.

Every function takes a single :class:`~orbitalgym.env.types.Trajectory`
(leaves with a leading time axis) and returns scalars, so callers batch
with ``jax.vmap``. Delta-v is accounted from the propellant trace through
the rocket equation when the side tracks mass, and from the commanded
delta-v otherwise.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from orbitalgym.actions.components import G0
from orbitalgym.games.proximity import lbg_events_from_positions, positions
from orbitalgym.rollout import episode_mask


class Outcome(IntEnum):
    TIMEOUT = 0
    CATCH = 1
    BREACH = 2
    BOTH = 3
    INVALID_IC = 4
    REPELLED = 5


@flax.struct.dataclass
class EpisodeMetrics:
    """Per-episode outcome and resource accounting.

    ``link_events_guard`` counts rising edges of the any-guard-in-contact
    signal within the episode mask (a link event starts when contact
    becomes true), not the number of masked steps in contact.
    ``in_cone_fraction_guard`` is the fraction of masked steps where any
    guard observation channel sees any bandit; 0 when the trajectory did
    not log a ``visible`` mask.

    The information fields describe what the guard team knows about the
    bandit nearest the lady, and are NaN when no guard belief log is
    supplied:

    ``belief_err_guard``
        Metres, time-averaged over the episode's live steps, of the
        smallest position error among the guards' belief means about that
        bandit.
    ``belief_err_guard_at_commit``
        The same quantity at the first live step where a bandit is inside
        ``commit_radius_m`` of the lady; NaN if no bandit ever commits.
    ``time_to_detect_guard``
        Seconds from the episode start to the first live step where that
        smallest error drops below ``detect_error_m``; NaN if it never
        does.
    ``belief_age_guard``
        Seconds since any guard last held a bandit in its sensor cone,
        time-averaged over the live steps.

    ``dwell_catch_steps`` and ``dwell_breach_steps`` are the longest dwell
    any bandit had run up at the terminating step: consecutive steps spent
    inside the catch radius of some guard, and inside the breach radius of
    the lady. They are reported whether or not the game's win conditions
    require a dwell, and are zero for a trajectory whose states carry no
    counters.
    """

    outcome: jax.Array
    steps: jax.Array
    min_d_gb: jax.Array
    min_d_bl: jax.Array
    dv_guard: jax.Array
    dv_bandit: jax.Array
    link_events_guard: jax.Array
    ic_valid: jax.Array
    in_cone_fraction_guard: jax.Array
    belief_err_guard: jax.Array
    belief_err_guard_at_commit: jax.Array
    time_to_detect_guard: jax.Array
    belief_age_guard: jax.Array
    dwell_catch_steps: jax.Array
    dwell_breach_steps: jax.Array


def _side_delta_v(
    side_state: Any,
    final_side_state: Any,
    params: Any,
    applied_dv: Any,
    mask: jax.Array,
    last_idx: jax.Array,
) -> jax.Array:
    """Team delta-v in m/s over the masked steps.

    With a propellant trace, delta-v follows from the rocket equation on
    the mass consumed between the state entering the first step and the
    state leaving the terminating step. The state leaving step ``k`` is the
    one entering step ``k + 1``, and for the last logged step it is
    ``final_side_state``, so an episode that fills the log still accounts
    for its final burn. Without a propellant trace, delta-v is the sum of
    the delta-v magnitudes the env actually imparted — thrust and an empty
    tank both clip a command, so the commanded magnitudes would overstate
    the spend.
    """
    if hasattr(side_state, "propellant_mass"):
        m_prop = side_state.propellant_mass  # (T, n)
        m_post = jnp.concatenate(
            [m_prop[1:], final_side_state.propellant_mass[None]], axis=0
        )  # (T, n) propellant leaving each step
        m0 = m_prop[0]
        m1 = m_post[last_idx]
        dry = jnp.asarray(params.dry_mass_kg, dtype=m_prop.dtype)
        per_vehicle = params.isp_s * G0 * jnp.log((dry + m0) / (dry + m1))
        return jnp.sum(per_vehicle)
    if applied_dv is None:
        raise ValueError(
            "delta-v accounting for a side without a propellant trace needs "
            "traj.applied_dv, the Δv the env actually imparted each step. "
            "Produce the trajectory with orbitalgym.rollout.rollout or "
            "belief_rollout."
        )
    dv_mag = jnp.linalg.norm(applied_dv, axis=-1)  # (T, n)
    return jnp.sum(dv_mag * mask[:, None])


@flax.struct.dataclass
class GuardInformation:
    """The four information metrics for one episode."""

    belief_err: jax.Array
    belief_err_at_commit: jax.Array
    time_to_detect: jax.Array
    belief_age: jax.Array


def _dwell_trace(traj: Any, name: str) -> jax.Array | None:
    """A dwell counter over the episode's steps, ``(T, n_b)``, or None.

    Entry ``k`` is the counter *leaving* step ``k``, so it lines up with the
    step's own catch and breach events. None when the trajectory's states
    carry no counter of that name, or carry an unsized one.
    """
    entering = getattr(traj.env_state, name, None)
    if entering is None or entering.shape[-1] == 0:
        return None
    leaving = getattr(traj.final_state, name)
    return jnp.concatenate([entering[1:], leaving[None]], axis=0)


def _require_dwell(trace: jax.Array | None) -> jax.Array:
    """A dwell counter trace, or an error naming what produces one."""
    if trace is None:
        raise ValueError(
            "the scenario's win conditions require a dwell, so classifying an "
            "episode needs the dwell counters on traj.env_state and "
            "traj.final_state. Produce the trajectory with "
            "orbitalgym.rollout.rollout or belief_rollout over an env built "
            "from this config."
        )
    return trace


def guard_information(
    traj: Any,
    belief_guard: Any,
    mask: jax.Array,
    steps: jax.Array,
    dt: float,
    commit_radius_m: float,
    detect_error_m: float,
) -> GuardInformation:
    """What the guard team knows about the bandit nearest the lady.

    ``belief_guard`` is the guard entry of the belief history that
    :func:`orbitalgym.rollout.belief_rollout` returns: leaves carry a
    leading time axis, so ``belief_guard.mean`` is ``(T, N_obs, N_total,
    d)``. Own-side entities occupy the first ``N_obs`` tracked slots and
    the bandits follow, and the first ``d // 2`` components of a tracked
    state are its position. The lady sits at the origin of the truth
    frame, so a bandit's distance to her is the norm of its position.
    """
    mean = belief_guard.mean  # (T, N_obs, N_total, d)
    n_obs = mean.shape[1]
    pos_dim = mean.shape[-1] // 2

    bandit_pos = positions(traj.env_state.bandits)[..., :pos_dim]  # (T, n_b, p)
    belief_bandit = mean[:, :, n_obs:, :pos_dim]  # (T, N_obs, n_b, p)

    d_bl = jnp.linalg.norm(bandit_pos, axis=-1)  # (T, n_b)
    nearest = jnp.argmin(d_bl, axis=-1)  # (T,)
    true_pos = jnp.take_along_axis(bandit_pos, nearest[:, None, None], axis=1)[:, 0]  # (T, p)
    guard_pos = jnp.take_along_axis(belief_bandit, nearest[:, None, None, None], axis=2)[:, :, 0]
    err = jnp.linalg.norm(guard_pos - true_pos[:, None, :], axis=-1)  # (T, N_obs)
    err_min = jnp.min(err, axis=-1)  # (T,)

    dtype = err_min.dtype
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    live = steps.astype(dtype)
    denom = jnp.maximum(live, 1.0)
    zero = jnp.zeros((), dtype=dtype)

    belief_err = jnp.sum(jnp.where(mask, err_min, zero)) / denom

    committed = mask & (jnp.min(d_bl, axis=-1) < commit_radius_m)
    commit_idx = jnp.argmax(committed)
    belief_err_at_commit = jnp.where(jnp.any(committed), err_min[commit_idx], nan)

    detected = mask & (err_min < detect_error_m)
    detect_idx = jnp.argmax(detected)
    time_to_detect = jnp.where(jnp.any(detected), detect_idx.astype(dtype) * dt, nan)

    visible = getattr(traj, "visible", None)
    if visible is None or visible.guard is None:
        belief_age = nan
    else:
        # Steps since a bandit was last in some guard's cone. The count is
        # taken over the whole prefix, so a bandit never yet seen leaves an
        # age of t + 1 at step t.
        idx = jnp.arange(mask.shape[0])
        seen_idx = jnp.where(jnp.any(visible.guard, axis=-1), idx, -1)
        age_steps = idx - jax.lax.cummax(seen_idx)
        belief_age = jnp.sum(jnp.where(mask, age_steps, 0)).astype(dtype) * dt / denom

    return GuardInformation(
        belief_err=belief_err,
        belief_err_at_commit=belief_err_at_commit,
        time_to_detect=time_to_detect,
        belief_age=belief_age,
    )


def lbg_episode_metrics(
    traj: Any,
    cfg: Any,
    *,
    belief_guard: Any = None,
    commit_radius_m: float = 2500.0,
    detect_error_m: float = 100.0,
) -> EpisodeMetrics:
    """Classify one LBG episode, account for its resources, and score what
    the guard team knew.

    ``belief_guard`` is the guard entry of the belief history returned by
    :func:`orbitalgym.rollout.belief_rollout`. Without it the information
    fields come back NaN.
    """
    if traj.final_state is None:
        raise ValueError(
            "lbg_episode_metrics needs traj.final_state: catch and breach are "
            "resolved over each step's transition, and without the state leaving "
            "the last step that step has no transition. Produce the trajectory "
            "with orbitalgym.rollout.rollout or belief_rollout."
        )

    mask = episode_mask(traj)  # (T,) True up to and including the terminating step
    steps = jnp.sum(mask.astype(jnp.int32))
    last_idx = steps - 1

    # Segment k runs from the state entering step k to the state leaving it,
    # so a T-step rollout has exactly T transitions and none is a state
    # paired with itself. Catch and breach are resolved over those segments
    # rather than at the sampled endpoints: a guard and a bandit can pass
    # within metres of each other between two decision samples.
    guard_pos = positions(traj.env_state.guards)  # (T, n_g, 3)
    bandit_pos = positions(traj.env_state.bandits)  # (T, n_b, 3)
    guard_final = positions(traj.final_state.guards)  # (n_g, 3)
    bandit_final = positions(traj.final_state.bandits)  # (n_b, 3)
    guard_next = jnp.concatenate([guard_pos[1:], guard_final[None]], axis=0)
    bandit_next = jnp.concatenate([bandit_pos[1:], bandit_final[None]], axis=0)

    catch_speed = float(getattr(cfg.game, "catch_speed_mps", jnp.inf))
    breach_speed = float(getattr(cfg.game, "breach_speed_mps", jnp.inf))
    caught_t, breached_t, d_gb_min_t, d_bl_min_t = lbg_events_from_positions(
        guard_pos,
        guard_next,
        bandit_pos,
        bandit_next,
        cfg.dt,
        cfg.game.catch_radius_m,
        catch_speed,
        cfg.game.breach_radius_m,
        breach_speed,
    )

    # The dwell counters take the same shift: the counter leaving step k is
    # the one entering step k + 1, and final_state closes the last step. A
    # dwell-gated win is read off them rather than off the step's own
    # closest approach, which is inside the radius on many steps that never
    # complete a dwell.
    dwell_catch_t = _dwell_trace(traj, "dwell_catch")
    dwell_breach_t = _dwell_trace(traj, "dwell_breach")
    catch_dwell_steps = int(getattr(cfg.game, "catch_dwell_steps", 0))
    breach_dwell_steps = int(getattr(cfg.game, "breach_dwell_steps", 0))
    if catch_dwell_steps > 0:
        caught_t = jnp.any(_require_dwell(dwell_catch_t) >= catch_dwell_steps, axis=-1)
    if breach_dwell_steps > 0:
        breached_t = jnp.any(_require_dwell(dwell_breach_t) >= breach_dwell_steps, axis=-1)

    big = jnp.asarray(jnp.inf, dtype=d_gb_min_t.dtype)
    min_d_gb = jnp.min(jnp.where(mask, d_gb_min_t, big))
    min_d_bl = jnp.min(jnp.where(mask, d_bl_min_t, big))

    # The event that ended the episode is the one on the last masked segment.
    T = mask.shape[0]  # noqa: N806
    caught = caught_t[last_idx]
    breached = breached_t[last_idx]
    # With a repel gate armed, an episode that stopped before the horizon
    # without a catch or a breach was ended by that gate: those are the only
    # terminal events the LBG termination function raises. With no gate armed
    # an early stop came from somewhere else — a composed drift cap, say — and
    # reads as a timeout, as does any episode that used every logged step or
    # every step the horizon allows.
    repel_armed = float(getattr(cfg.game, "escape_radius_m", 0.0)) > 0.0 or bool(
        getattr(cfg.game, "repel_on_empty_tank", False)
    )
    if repel_armed:
        quiet = jnp.where(steps < min(T, cfg.max_steps), Outcome.REPELLED, Outcome.TIMEOUT)
    else:
        quiet = jnp.asarray(Outcome.TIMEOUT)
    outcome = jnp.where(
        caught & breached,
        Outcome.BOTH,
        jnp.where(caught, Outcome.CATCH, jnp.where(breached, Outcome.BREACH, quiet)),
    )
    ic_valid = traj.env_state.ic_valid[0]
    outcome = jnp.where(ic_valid, outcome, Outcome.INVALID_IC).astype(jnp.int32)

    applied_dv = getattr(traj, "applied_dv", None)
    dv_guard = _side_delta_v(
        traj.env_state.guards,
        traj.final_state.guards,
        cfg.guard_params,
        None if applied_dv is None else applied_dv.guard,
        mask,
        last_idx,
    )
    dv_bandit = _side_delta_v(
        traj.env_state.bandits,
        traj.final_state.bandits,
        cfg.bandit_params,
        None if applied_dv is None else applied_dv.bandit,
        mask,
        last_idx,
    )

    contact = getattr(traj, "contact", None)
    if contact is None or contact.guard is None:
        link_events = jnp.asarray(0, dtype=jnp.int32)
    else:
        any_contact = jnp.any(contact.guard, axis=-1)  # (T,) any guard in contact this step
        # Rising edges of the any-guard-in-contact signal within the episode
        # mask: a link event starts when contact becomes true after a masked
        # step where it was false (or at the first masked step).
        prev_contact = jnp.concatenate([jnp.zeros((1,), dtype=bool), any_contact[:-1]])
        rising = any_contact & (~prev_contact) & mask
        link_events = jnp.sum(rising.astype(jnp.int32))

    visible = getattr(traj, "visible", None)
    if visible is None or visible.guard is None:
        in_cone_fraction_guard = jnp.asarray(0.0)
    else:
        any_visible = jnp.any(visible.guard, axis=-1)  # (T,) any guard sees any bandit
        denom = jnp.maximum(steps, 1)
        in_cone_fraction_guard = jnp.sum((any_visible & mask).astype(jnp.float32)) / denom.astype(
            jnp.float32
        )

    if belief_guard is None:
        nan = jnp.asarray(jnp.nan)
        info = GuardInformation(
            belief_err=nan, belief_err_at_commit=nan, time_to_detect=nan, belief_age=nan
        )
    else:
        info = guard_information(
            traj, belief_guard, mask, steps, cfg.dt, commit_radius_m, detect_error_m
        )

    zero = jnp.asarray(0, dtype=jnp.int32)
    dwell_catch_steps = (
        zero if dwell_catch_t is None else jnp.max(dwell_catch_t[last_idx]).astype(jnp.int32)
    )
    dwell_breach_steps = (
        zero if dwell_breach_t is None else jnp.max(dwell_breach_t[last_idx]).astype(jnp.int32)
    )

    return EpisodeMetrics(
        outcome=outcome,
        steps=steps.astype(jnp.int32),
        min_d_gb=min_d_gb,
        min_d_bl=min_d_bl,
        dv_guard=dv_guard,
        dv_bandit=dv_bandit,
        link_events_guard=link_events,
        ic_valid=ic_valid,
        in_cone_fraction_guard=in_cone_fraction_guard,
        belief_err_guard=info.belief_err,
        belief_err_guard_at_commit=info.belief_err_at_commit,
        time_to_detect_guard=info.time_to_detect,
        belief_age_guard=info.belief_age,
        dwell_catch_steps=dwell_catch_steps,
        dwell_breach_steps=dwell_breach_steps,
    )
