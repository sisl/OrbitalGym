"""Reference reward: negative sum of guard distances to the reference orbit origin.

Side-aware: returns the negative-distance sum for the GUARD side and 0.0 for
the BANDIT side. This preserves the pre-Phase-1 single-agent numerical
behavior (where only the guard had a reward signal).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from orbital_game.env.types import Side
from orbital_game.registry import RewardFnKey, register
from orbital_game.rewards.base import RewardScope


@register(RewardFnKey.DISTANCE_TO_REFERENCE_ORBIT)
@dataclass(frozen=True)
class DistanceToReferenceOrbit:
    """Negative sum of guard-to-reference-orbit-origin distances.

    Returns 0.0 for the bandit side (preserving the bootstrap's
    single-reward-on-guard-only behavior). Future zero-sum games negate
    explicitly inside their own reward classes.
    """

    scope: RewardScope = RewardScope.PER_SIDE

    def __call__(self, prev_state, action, next_state, side, params, t):
        del prev_state, action, params, t
        if side is Side.BANDIT:
            return jnp.asarray(0.0)
        guards = next_state.guards
        positions = guards.rtn[:, :3] if hasattr(guards, "rtn") else guards.rt[:, :2]
        distances = jnp.linalg.norm(positions, axis=-1)
        return -jnp.sum(distances)
