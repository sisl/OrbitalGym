"""Reference reward: negative sum of guard distances to the reference orbit origin.

The reference orbit sits at the origin of the RTN (or RT) frame, so each guard's
distance is the norm of its position components — 3D for RTN, 2D for RT.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from orbital_game.registry import RewardFnKey, register


@register(RewardFnKey.DISTANCE_TO_REFERENCE_ORBIT)
@dataclass(frozen=True)
class DistanceToReferenceOrbit:
    """Negative sum of guard-to-reference-orbit-origin distances.

    Works with either dynamics choice: RTN state → 3D norm over (r, t, n);
    RT state → 2D norm over (r, t).
    """

    def __call__(self, prev_state, action, next_state, params, t):
        del prev_state, action, params, t
        guards = next_state.guards
        # RTN state → 3D norm over (r, t, n); RT state → 2D norm over (r, t).
        positions = guards.rtn[:, :3] if hasattr(guards, "rtn") else guards.rt[:, :2]
        distances = jnp.linalg.norm(positions, axis=-1)
        return -jnp.sum(distances)
