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
from orbitalgym.rollout import episode_mask


class Outcome(IntEnum):
    TIMEOUT = 0
    CATCH = 1
    BREACH = 2
    BOTH = 3
    INVALID_IC = 4


@flax.struct.dataclass
class EpisodeMetrics:
    outcome: jax.Array
    steps: jax.Array
    min_d_gb: jax.Array
    min_d_bl: jax.Array
    dv_guard: jax.Array
    dv_bandit: jax.Array
    link_events_guard: jax.Array
    ic_valid: jax.Array


def _positions(side_state: Any) -> jax.Array:
    """Per-step per-vehicle positions, shape (T, n, dim)."""
    if hasattr(side_state, "rtn"):
        return side_state.rtn[..., :3]
    return side_state.rt[..., :2]


def _side_delta_v(side_traj: Any, side_state: Any, params: Any, mask: jax.Array) -> jax.Array:
    """Team delta-v in m/s over the masked steps.

    With a propellant trace, delta-v follows from the rocket equation on
    the mass consumed between the first and last valid step. Without one,
    it is the sum of commanded delta-v magnitudes.
    """
    if hasattr(side_state, "propellant_mass"):
        m_prop = side_state.propellant_mass  # (T, n)
        m0 = m_prop[0]
        last_idx = jnp.sum(mask.astype(jnp.int32)) - 1
        m1 = m_prop[last_idx]
        dry = jnp.asarray(params.dry_mass_kg, dtype=m_prop.dtype)
        per_vehicle = params.isp_s * G0 * jnp.log((dry + m0) / (dry + m1))
        return jnp.sum(per_vehicle)
    dv_mag = jnp.linalg.norm(side_traj.action.dv, axis=-1)  # (T, n)
    return jnp.sum(dv_mag * mask[:, None])


def lbg_episode_metrics(traj: Any, cfg: Any) -> EpisodeMetrics:
    """Classify one LBG episode and account for its resources."""
    mask = episode_mask(traj)  # (T,) True up to and including the terminating step
    steps = jnp.sum(mask.astype(jnp.int32))
    last_idx = steps - 1

    guard_pos = _positions(traj.env_state.guards)  # (T, n_g, dim)
    bandit_pos = _positions(traj.env_state.bandits)  # (T, n_b, dim)
    d_gb = jnp.linalg.norm(guard_pos[:, :, None, :] - bandit_pos[:, None, :, :], axis=-1)
    d_gb_min_t = jnp.min(d_gb, axis=(1, 2))  # (T,)
    d_bl_min_t = jnp.min(jnp.linalg.norm(bandit_pos, axis=-1), axis=1)  # (T,)

    big = jnp.asarray(jnp.inf, dtype=d_gb_min_t.dtype)
    min_d_gb = jnp.min(jnp.where(mask, d_gb_min_t, big))
    min_d_bl = jnp.min(jnp.where(mask, d_bl_min_t, big))

    # The logged env_state at index k is the state entering step k; the
    # terminating event is visible in the state after the last valid step,
    # which the freeze-on-done rule holds at index last_idx + 1 when it
    # exists. Use the post-step state when available, else the last state.
    T = mask.shape[0]  # noqa: N806
    post_idx = jnp.minimum(last_idx + 1, T - 1)
    terminated = traj.episode_done[last_idx]
    d_gb_final = jnp.where(terminated, d_gb_min_t[post_idx], d_gb_min_t[last_idx])
    d_bl_final = jnp.where(terminated, d_bl_min_t[post_idx], d_bl_min_t[last_idx])
    min_d_gb = jnp.minimum(min_d_gb, jnp.where(terminated, d_gb_final, big))
    min_d_bl = jnp.minimum(min_d_bl, jnp.where(terminated, d_bl_final, big))

    caught = d_gb_final < cfg.game.catch_radius_m
    breached = d_bl_final < cfg.game.breach_radius_m
    outcome = jnp.where(
        caught & breached,
        Outcome.BOTH,
        jnp.where(caught, Outcome.CATCH, jnp.where(breached, Outcome.BREACH, Outcome.TIMEOUT)),
    )
    ic_valid = traj.env_state.ic_valid[0]
    outcome = jnp.where(ic_valid, outcome, Outcome.INVALID_IC).astype(jnp.int32)

    dv_guard = _side_delta_v(traj.sides.guard, traj.env_state.guards, cfg.guard_params, mask)
    dv_bandit = _side_delta_v(traj.sides.bandit, traj.env_state.bandits, cfg.bandit_params, mask)

    contact = getattr(traj, "contact", None)
    if contact is None or contact.guard is None:
        link_events = jnp.asarray(0, dtype=jnp.int32)
    else:
        any_contact = jnp.any(contact.guard, axis=-1)  # (T,)
        link_events = jnp.sum((any_contact & mask).astype(jnp.int32))

    return EpisodeMetrics(
        outcome=outcome,
        steps=steps.astype(jnp.int32),
        min_d_gb=min_d_gb,
        min_d_bl=min_d_bl,
        dv_guard=dv_guard,
        dv_bandit=dv_bandit,
        link_events_guard=link_events,
        ic_valid=ic_valid,
    )
