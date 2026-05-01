"""Post-sample combined-state validators for the IC rejection loop.

Each validator is a frozen dataclass with __call__(config, defs, ints) returning
a scalar bool jax.Array. The env's reset loop ANDs together all validators
in the ICSpec; one False triggers re-sampling. Validators must be vmap-safe
(no Python control flow on traced values).

Position is read from `defs.rtn[:, :3]` / `ints.rtn[:, :3]` (or `defs.rt[:, :2]`
for 2D scenarios). The validator inspects whichever exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import jax.numpy as jnp

from orbital_game.registry import ValidatorKey, register


class SeparationScope(StrEnum):
    ALL = "all"
    WITHIN_DEFENDERS = "within_defenders"
    WITHIN_INTRUDERS = "within_intruders"
    CROSS = "cross"


class SideSelector(StrEnum):
    ALL = "all"
    DEFENDERS = "defenders"
    INTRUDERS = "intruders"


def _positions(side) -> jnp.ndarray:
    """Extract (N, 3) positions from a side state, padding 2D RT to 3D."""
    if hasattr(side, "rtn"):
        return side.rtn[:, :3]
    if hasattr(side, "rt"):
        n = side.rt.shape[0]
        z = jnp.zeros((n, 1))
        return jnp.concatenate([side.rt[:, :2], z], axis=-1)
    raise AttributeError("side state has neither rt nor rtn")


def _min_pairwise_distance(p: jnp.ndarray) -> jnp.ndarray:
    """Min pairwise distance in (N, 3) positions. Returns +inf for N<2."""
    n = p.shape[0]
    if n < 2:
        return jnp.asarray(jnp.inf)
    diff = p[:, None, :] - p[None, :, :]
    sq = jnp.sum(diff * diff, axis=-1)
    sq = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, sq)
    return jnp.sqrt(jnp.min(sq))


def _min_cross_distance(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Min distance between any point in a (Na, 3) and any in b (Nb, 3)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return jnp.asarray(jnp.inf)
    diff = a[:, None, :] - b[None, :, :]
    sq = jnp.sum(diff * diff, axis=-1)
    return jnp.sqrt(jnp.min(sq))


@register(ValidatorKey.MIN_SEPARATION)
@dataclass(frozen=True)
class MinSeparation:
    """Reject if the configured pair scope has any pair closer than distance_m.

    scope:
      ALL - every pair across both sides
      WITHIN_DEFENDERS - defender-defender pairs only
      WITHIN_INTRUDERS - intruder-intruder pairs only
      CROSS - defender-intruder pairs only
    """

    distance_m: float
    scope: SeparationScope = SeparationScope.ALL

    def __call__(self, config, defenders, intruders) -> jnp.ndarray:
        del config
        pd = _positions(defenders)
        pi = _positions(intruders)
        if self.scope == SeparationScope.WITHIN_DEFENDERS:
            d = _min_pairwise_distance(pd)
        elif self.scope == SeparationScope.WITHIN_INTRUDERS:
            d = _min_pairwise_distance(pi)
        elif self.scope == SeparationScope.CROSS:
            d = _min_cross_distance(pd, pi)
        else:  # ALL
            all_p = jnp.concatenate([pd, pi], axis=0)
            d = _min_pairwise_distance(all_p)
        return d >= self.distance_m


@register(ValidatorKey.MAX_RANGE)
@dataclass(frozen=True)
class MaxRange:
    """Reject if any selected vehicle is farther than distance_m from HVA origin.

    Since RTN positions are HVA-relative, the HVA origin is (0,0,0) in this frame.
    side:
      ALL - both defenders and intruders
      DEFENDERS - defenders only
      INTRUDERS - intruders only
    """

    distance_m: float
    side: SideSelector = SideSelector.ALL

    def __call__(self, config, defenders, intruders) -> jnp.ndarray:
        del config
        if self.side == SideSelector.DEFENDERS:
            p = _positions(defenders)
        elif self.side == SideSelector.INTRUDERS:
            p = _positions(intruders)
        else:
            p = jnp.concatenate([_positions(defenders), _positions(intruders)], axis=0)
        ranges = jnp.linalg.norm(p, axis=-1)
        return jnp.all(ranges <= self.distance_m)
