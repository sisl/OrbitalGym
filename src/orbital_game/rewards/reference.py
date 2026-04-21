"""Reference reward: negative sum of defender distances to the HVA.

The HVA sits at the origin of the RTN (or RT) frame, so each defender's
distance is the norm of its position components — 3D for RTN, 2D for RT.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from orbital_game.registry import RewardFnKey, register


@register(RewardFnKey.DISTANCE_TO_HVA)
@dataclass(frozen=True)
class DistanceToHVA:
    """Negative sum of defender-to-HVA distances. Works with either dynamics choice:
    RTN state → 3D norm over (r, t, n); RT state → 2D norm over (r, t).
    """

    def __call__(self, prev_state, action, next_state, params, t):
        del prev_state, action, params, t
        defs = next_state.defenders
        # RTN state → 3D norm over (r, t, n); RT state → 2D norm over (r, t).
        positions = defs.rtn[:, :3] if hasattr(defs, "rtn") else defs.rt[:, :2]
        distances = jnp.linalg.norm(positions, axis=-1)
        return -jnp.sum(distances)
