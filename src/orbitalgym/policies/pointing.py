"""PointingPolicy: fill the PointAt target from the agent's belief."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.registry import PolicyKey, register


class PointingTarget(StrEnum):
    HOLD = "hold"
    LADY = "lady"
    OPPONENT_BELIEF = "opponent_belief"
    TEAMMATE = "teammate"


def _pad3(v: jax.Array) -> jax.Array:
    if v.shape[-1] == 3:
        return v
    return jnp.concatenate([v, jnp.zeros(v.shape[:-1] + (1,), dtype=v.dtype)], axis=-1)


def _unit(v: jax.Array) -> jax.Array:
    return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def pointing_direction(mean: jax.Array, target: PointingTarget) -> jax.Array:
    """Compute the ``target_dir`` (n_self, 3) unit vectors for a pointing target.

    ``mean`` has shape ``(N_obs, N_total, d)``: rows are this side's
    observers, columns are own-side vehicles first then the opposing side.
    Observer ``i`` reads its own position from cell ``(i, i)``.
    """
    n_self, n_total, d = mean.shape
    del n_total
    pos_dim = 2 if d == 4 else 3
    idx = jnp.arange(n_self)
    own = mean[idx, idx, :pos_dim]  # (n_self, pos_dim)

    if target is PointingTarget.HOLD:
        return jnp.zeros((n_self, 3), dtype=mean.dtype)
    if target is PointingTarget.LADY:
        return _unit(_pad3(-own))
    if target is PointingTarget.OPPONENT_BELIEF:
        opp = mean[:, n_self:, :pos_dim]  # (n_self, n_opp, pos_dim)
        diff = opp - own[:, None, :]
        nearest = jnp.argmin(jnp.linalg.norm(diff, axis=-1), axis=1)
        return _unit(_pad3(diff[idx, nearest]))
    team = mean[:, :n_self, :pos_dim]
    diff = team - own[:, None, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    dist = jnp.where(jnp.eye(n_self, dtype=bool), jnp.inf, dist)
    nearest = jnp.argmin(dist, axis=1)
    chosen = _pad3(diff[idx, nearest])
    return jnp.where(n_self > 1, _unit(chosen), jnp.zeros_like(chosen))


@register(PolicyKey.POINTING)
@dataclass(frozen=True)
class PointingPolicy:
    """Wrap a delta-v policy and set ``target_dir`` from the belief mean.

    The belief mean has shape ``(N_obs, N_total, d)``: rows are this side's
    observers, columns are own-side vehicles first then the opposing side.
    Observer ``i`` reads its own position from cell ``(i, i)``.
    """

    inner: Any
    target: PointingTarget

    def __call__(self, policy_state: Any, agent_view: Any, key: jax.Array, t: jax.Array):
        cmd, next_state = self.inner(policy_state, agent_view, key, t)
        direction = pointing_direction(agent_view.mean, self.target)
        return cmd.replace(target_dir=direction.astype(cmd.target_dir.dtype)), next_state
