"""Reference reward: negative sum of defender distances to the HVA.

The HVA sits at the origin of the RTN frame, so each defender's distance is
simply the norm of its rtn position components.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from orbital_game.registry import RewardFnKey, register


@register(RewardFnKey.DISTANCE_TO_HVA)
@dataclass(frozen=True)
class DistanceToHVA:
    def __call__(self, prev_state, action, next_state, params, t):
        del prev_state, action, params, t
        # Defender rtn[:, :3] is position; negate sum of norms.
        positions = next_state.defenders.rtn[:, :3]
        distances = jnp.linalg.norm(positions, axis=-1)
        return -jnp.sum(distances)
