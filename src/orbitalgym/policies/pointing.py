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
    OPPONENT_SCAN = "opponent_scan"
    TEAMMATE = "teammate"


def _pad3(v: jax.Array) -> jax.Array:
    if v.shape[-1] == 3:
        return v
    return jnp.concatenate([v, jnp.zeros(v.shape[:-1] + (1,), dtype=v.dtype)], axis=-1)


def _unit(v: jax.Array) -> jax.Array:
    return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _pos_dim(d: int) -> int:
    return 2 if d == 4 else 3


def _nearest_opponent(mean: jax.Array, n_self: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(idx, own, nearest)``: observer indices, own positions, nearest columns."""
    pos_dim = _pos_dim(mean.shape[-1])
    idx = jnp.arange(n_self)
    own = mean[idx, idx, :pos_dim]  # (n_self, pos_dim)
    opp = mean[:, n_self:, :pos_dim]  # (n_self, n_opp, pos_dim)
    nearest = jnp.argmin(jnp.linalg.norm(opp - own[:, None, :], axis=-1), axis=1)
    return idx, own, nearest


def _belief_attr(agent_view: Any, name: str) -> Any:
    """Read ``name`` off the belief, following ``inner`` through any wrappers."""
    node: Any = agent_view
    while node is not None:
        value = getattr(node, name, None)
        if value is not None:
            return value
        node = getattr(node, "inner", None)
    return None


def opponent_spread(agent_view: Any, n_self: int) -> jax.Array:
    """Positional uncertainty (metres) about the nearest opponent, per observer.

    Particle beliefs report the RMS distance of that opponent's particle
    cloud from the belief mean, which is the weight-carrying estimate the
    rest of the policy points at; Gaussian beliefs report the square
    root of the trace of the position block of its covariance; a belief
    carrying only a mean reports zero. Belief wrappers are followed
    through their ``inner`` field. The branch is taken on Python
    attribute presence, so the result is jit-safe.
    """
    mean = agent_view.mean
    pos_dim = _pos_dim(mean.shape[-1])
    idx, _, nearest = _nearest_opponent(mean, n_self)

    particles = _belief_attr(agent_view, "particles")
    if particles is not None:
        cloud = particles[:, n_self:, :, :pos_dim][idx, nearest]  # (n_self, K, pos_dim)
        centre = mean[:, n_self:, :pos_dim][idx, nearest]  # (n_self, pos_dim)
        centered = cloud - centre[:, None, :]
        return jnp.sqrt(jnp.mean(jnp.sum(centered**2, axis=-1), axis=-1))

    cov = _belief_attr(agent_view, "cov")
    if cov is not None:
        block = cov[:, n_self:, :pos_dim, :pos_dim][idx, nearest]  # (n_self, pos_dim, pos_dim)
        return jnp.sqrt(jnp.trace(block, axis1=-2, axis2=-1))

    return jnp.zeros((n_self,), dtype=mean.dtype)


def pointing_direction(
    mean: jax.Array,
    target: PointingTarget,
    *,
    t: jax.Array | float | None = None,
    spread: jax.Array | None = None,
    scan_spread_m: float = 200.0,
    scan_rate_rad_s: float = 0.0175,
) -> jax.Array:
    """Compute the ``target_dir`` (n_self, 3) unit vectors for a pointing target.

    ``mean`` has shape ``(N_obs, N_total, d)``: rows are this side's
    observers, columns are own-side vehicles first then the opposing side.
    Observer ``i`` reads its own position from cell ``(i, i)``.

    ``PointingTarget.OPPONENT_SCAN`` additionally needs the elapsed time
    ``t`` and the per-observer opponent ``spread`` from `opponent_spread`.
    """
    n_self, n_total, d = mean.shape
    del n_total
    pos_dim = _pos_dim(d)
    idx = jnp.arange(n_self)
    own = mean[idx, idx, :pos_dim]  # (n_self, pos_dim)

    if target is PointingTarget.HOLD:
        return jnp.zeros((n_self, 3), dtype=mean.dtype)
    if target is PointingTarget.LADY:
        return _unit(_pad3(-own))
    if target in (PointingTarget.OPPONENT_BELIEF, PointingTarget.OPPONENT_SCAN):
        opp = mean[:, n_self:, :pos_dim]  # (n_self, n_opp, pos_dim)
        diff = opp - own[:, None, :]
        nearest = jnp.argmin(jnp.linalg.norm(diff, axis=-1), axis=1)
        belief_dir = _unit(_pad3(diff[idx, nearest]))
        if target is PointingTarget.OPPONENT_BELIEF:
            return belief_dir
        if t is None or spread is None:
            raise ValueError("PointingTarget.OPPONENT_SCAN requires both `t` and `spread`")
        phase = 2.0 * jnp.pi * idx / n_self
        az = jnp.asarray(scan_rate_rad_s, dtype=mean.dtype) * jnp.asarray(t, dtype=mean.dtype)
        az = az + phase.astype(mean.dtype)
        scan_dir = jnp.stack([jnp.cos(az), jnp.sin(az), jnp.zeros_like(az)], axis=-1)
        scanning = jnp.asarray(spread, dtype=mean.dtype) > scan_spread_m
        return jnp.where(scanning[:, None], scan_dir, belief_dir)
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

    Under `PointingTarget.OPPONENT_SCAN` an observer whose opponent belief
    is spread wider than ``scan_spread_m`` sweeps its boresight through the
    radial/along-track plane at ``scan_rate_rad_s``, staggered against its
    teammates, and falls back to the belief mean once the spread tightens.
    """

    inner: Any
    target: PointingTarget
    scan_spread_m: float = 200.0
    scan_rate_rad_s: float = 0.0175

    def __call__(self, policy_state: Any, agent_view: Any, key: jax.Array, t: jax.Array):
        cmd, next_state = self.inner(policy_state, agent_view, key, t)
        spread = None
        if self.target is PointingTarget.OPPONENT_SCAN:
            spread = opponent_spread(agent_view, agent_view.mean.shape[0])
        direction = pointing_direction(
            agent_view.mean,
            self.target,
            t=t,
            spread=spread,
            scan_spread_m=self.scan_spread_m,
            scan_rate_rad_s=self.scan_rate_rad_s,
        )
        return cmd.replace(target_dir=direction.astype(cmd.target_dir.dtype)), next_state
