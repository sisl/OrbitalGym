"""Team link predicates.

A link answers, per agent and per tick, whether that agent can exchange
information with its team. ``belief_rollout`` uses the mask to gate team
belief fusion and records it on the trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.dynamics.quaternion import quat_to_rotation_matrix
from orbitalgym.env.types import Side
from orbitalgym.groundstations.contacts import in_contact_now
from orbitalgym.registry import LinkKey, register


def _side_state(env_state: Any, side: Side) -> Any:
    return env_state.guards if side is Side.GUARD else env_state.bandits


def _positions_3d(side_state: Any) -> jax.Array:
    if hasattr(side_state, "rtn"):
        return side_state.rtn[:, :3]
    pos = side_state.rt[:, :2]
    return jnp.concatenate([pos, jnp.zeros((pos.shape[0], 1), dtype=pos.dtype)], axis=1)


def _quats(side_state: Any, n: int, dtype) -> jax.Array:
    if hasattr(side_state, "quat"):
        return side_state.quat
    return jnp.zeros((n, 4), dtype=dtype).at[:, 0].set(1.0)


@register(LinkKey.ALWAYS)
@dataclass(frozen=True)
class AlwaysLinked:
    """Every agent is linked every tick (centralized regime)."""

    def __call__(self, env_state: Any, side: Side, t: jax.Array) -> jax.Array:
        n = _positions_3d(_side_state(env_state, side)).shape[0]
        return jnp.ones((n,), dtype=jnp.bool_)


@register(LinkKey.GROUND_NETWORK)
@dataclass(frozen=True)
class GroundNetworkLink:
    """Linked while the reference orbit is inside any ground-station window."""

    schedule: Any

    def __call__(self, env_state: Any, side: Side, t: jax.Array) -> jax.Array:
        n = _positions_3d(_side_state(env_state, side)).shape[0]
        return jnp.broadcast_to(in_contact_now(self.schedule, t), (n,))


@register(LinkKey.POINTING_CONE)
@dataclass(frozen=True)
class PointingConeLink:
    """Linked when a teammate lies inside this agent's body-fixed communication cone.

    With ``mutual=True`` both agents must hold each other in their cones, which
    models a laser crosslink; with ``mutual=False`` a one-sided beam suffices.
    An agent with no teammates is never linked. Sides without an attitude state
    use the identity quaternion.
    """

    half_angle_rad: float
    boresight_body: tuple[float, float, float] = (0.0, 1.0, 0.0)
    mutual: bool = True

    def __call__(self, env_state: Any, side: Side, t: jax.Array) -> jax.Array:
        del t
        side_state = _side_state(env_state, side)
        pos = _positions_3d(side_state)  # (n, 3)
        n = pos.shape[0]
        dtype = pos.dtype
        quat = _quats(side_state, n, dtype)
        rot = jax.vmap(quat_to_rotation_matrix)(quat)  # (n, 3, 3)
        b_body = jnp.asarray(self.boresight_body, dtype=dtype)
        b_world = rot @ (b_body / jnp.linalg.norm(b_body))  # (n, 3)

        diff = pos[None, :, :] - pos[:, None, :]  # (i, j, 3): from i to j
        dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        los = diff / jnp.maximum(dist, jnp.asarray(1e-12, dtype=dtype))
        cos_theta = jnp.einsum("ik,ijk->ij", b_world, los)
        in_cone = cos_theta >= jnp.cos(jnp.asarray(self.half_angle_rad, dtype=dtype))
        not_self = ~jnp.eye(n, dtype=jnp.bool_)
        pair = in_cone & not_self
        if self.mutual:
            pair = pair & pair.T
        return jnp.any(pair, axis=1)


@register(LinkKey.DISTANCE)
@dataclass(frozen=True)
class DistanceLink:
    """Symmetric distance contact for a team of at most two agents.

    Both agents participate iff their separation is <= ``max_range_m`` metres.
    A singleton or empty team is never linked. Teams larger than two raise:
    the rollout's per-agent mask pools all linked agents into one fusion group,
    and cannot represent separate connected components. Three-agent and larger
    communication topologies require graph-aware fusion, not this predicate.
    This gate models distance only, without attitude or line-of-sight blockage.
    """

    max_range_m: float

    def __post_init__(self):
        try:
            radius = float(self.max_range_m)
        except (TypeError, ValueError) as error:
            raise ValueError("max_range_m must be finite positive metres") from error
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("max_range_m must be finite positive metres")
        object.__setattr__(self, "max_range_m", radius)

    def __call__(self, env_state: Any, side: Side, t: jax.Array) -> jax.Array:
        del t
        pos = _positions_3d(_side_state(env_state, side))
        n = pos.shape[0]
        if n > 2:
            raise ValueError(
                "DistanceLink supports at most two agents; larger teams need graph-aware fusion"
            )
        if n < 2:
            return jnp.zeros((n,), dtype=jnp.bool_)
        linked = jnp.linalg.norm(pos[1] - pos[0]) <= self.max_range_m
        return jnp.broadcast_to(linked, (2,))
