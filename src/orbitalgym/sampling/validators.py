"""Post-sample combined-state validators for the IC rejection loop.

Each validator is a frozen dataclass with __call__(config, guards, bandits) returning
a scalar bool jax.Array. The env's reset loop ANDs together all validators
in the ICSpec; one False triggers re-sampling. Validators must be vmap-safe
(no Python control flow on traced values).

Position is read from `guards.rtn[:, :3]` / `bandits.rtn[:, :3]` (or `guards.rt[:, :2]`
for 2D scenarios). The validator inspects whichever exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import jax.numpy as jnp

from orbitalgym.registry import ValidatorKey, register


class SeparationScope(StrEnum):
    ALL = "all"
    WITHIN_GUARDS = "within_guards"
    WITHIN_BANDITS = "within_bandits"
    CROSS = "cross"


class SideSelector(StrEnum):
    ALL = "all"
    GUARDS = "guards"
    BANDITS = "bandits"


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
      WITHIN_GUARDS - guard-guard pairs only
      WITHIN_BANDITS - bandit-bandit pairs only
      CROSS - guard-bandit pairs only
    """

    distance_m: float
    scope: SeparationScope = SeparationScope.ALL

    def __call__(self, config, guards, bandits) -> jnp.ndarray:
        del config
        pg = _positions(guards)
        pb = _positions(bandits)
        if self.scope == SeparationScope.WITHIN_GUARDS:
            d = _min_pairwise_distance(pg)
        elif self.scope == SeparationScope.WITHIN_BANDITS:
            d = _min_pairwise_distance(pb)
        elif self.scope == SeparationScope.CROSS:
            d = _min_cross_distance(pg, pb)
        else:  # ALL
            all_p = jnp.concatenate([pg, pb], axis=0)
            d = _min_pairwise_distance(all_p)
        return d >= self.distance_m


@register(ValidatorKey.MAX_RANGE)
@dataclass(frozen=True)
class MaxRange:
    """Reject if any selected vehicle is farther than distance_m from the reference orbit origin.

    Since RTN positions are reference-orbit-relative, the reference orbit origin
    is (0,0,0) in this frame.
    side:
      ALL - both guards and bandits
      GUARDS - guards only
      BANDITS - bandits only
    """

    distance_m: float
    side: SideSelector = SideSelector.ALL

    def __call__(self, config, guards, bandits) -> jnp.ndarray:
        del config
        if self.side == SideSelector.GUARDS:
            p = _positions(guards)
        elif self.side == SideSelector.BANDITS:
            p = _positions(bandits)
        else:
            p = jnp.concatenate([_positions(guards), _positions(bandits)], axis=0)
        ranges = jnp.linalg.norm(p, axis=-1)
        return jnp.all(ranges <= self.distance_m)
