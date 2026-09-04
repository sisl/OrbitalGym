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


def _side_delta_v(
    side_traj: Any, side_state: Any, params: Any, mask: jax.Array, post_idx: jax.Array
) -> jax.Array:
    """Team delta-v in m/s over the masked steps.

    With a propellant trace, delta-v follows from the rocket equation on
    the mass consumed between the first step and the post-terminal state
    at ``post_idx``. Without one, it is the sum of commanded delta-v
    magnitudes.
    """
    if hasattr(side_state, "propellant_mass"):
        m_prop = side_state.propellant_mass  # (T, n)
        m0 = m_prop[0]
        m1 = m_prop[post_idx]
        dry = jnp.asarray(params.dry_mass_kg, dtype=m_prop.dtype)
        per_vehicle = params.isp_s * G0 * jnp.log((dry + m0) / (dry + m1))
        return jnp.sum(per_vehicle)
    dv_mag = jnp.linalg.norm(side_traj.action.dv, axis=-1)  # (T, n)
    return jnp.sum(dv_mag * mask[:, None])


def lbg_episode_metrics(traj: Any, cfg: Any) -> EpisodeMetrics:
    """Classify one LBG episode and account for its resources."""
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

    big = jnp.asarray(jnp.inf, dtype=d_gb_min_t.dtype)
    min_d_gb = jnp.min(jnp.where(mask, d_gb_min_t, big))
    min_d_bl = jnp.min(jnp.where(mask, d_bl_min_t, big))

    # The event that ended the episode is the one on the last masked segment.
    T = mask.shape[0]  # noqa: N806
    post_idx = jnp.minimum(last_idx + 1, T - 1)
    caught = caught_t[last_idx]
    breached = breached_t[last_idx]
    # An episode that stopped before the horizon without a catch or a breach
    # was ended by the repel gate: those are the only terminal events the LBG
    # termination function raises. An episode that used every logged step, or
    # every step the horizon allows, timed out instead.
    stopped_early = steps < min(T, cfg.max_steps)
    quiet = jnp.where(stopped_early, Outcome.REPELLED, Outcome.TIMEOUT)
    outcome = jnp.where(
        caught & breached,
        Outcome.BOTH,
        jnp.where(caught, Outcome.CATCH, jnp.where(breached, Outcome.BREACH, quiet)),
    )
    ic_valid = traj.env_state.ic_valid[0]
    outcome = jnp.where(ic_valid, outcome, Outcome.INVALID_IC).astype(jnp.int32)

    dv_guard = _side_delta_v(
        traj.sides.guard, traj.env_state.guards, cfg.guard_params, mask, post_idx
    )
    dv_bandit = _side_delta_v(
        traj.sides.bandit, traj.env_state.bandits, cfg.bandit_params, mask, post_idx
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
    )
