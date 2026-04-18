"""Reference termination: episode ends on max_steps OR defender breach radius."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.registry import TerminationFnKey, register


@register(TerminationFnKey.MAX_STEPS_OR_BREACH)
@dataclass(frozen=True)
class MaxStepsOrBreach:
    """Terminate when step count hits max_steps OR when any defender is inside
    breach_distance_m of the HVA origin.

    Assumes the scenario uses RTNState (reads `state.defenders.rtn[:, :3]` for
    position). For RTState (2D) scenarios this would need to read `rt[:, :2]`.
    """

    max_steps: int
    breach_distance_m: float

    def __call__(self, state, params, t) -> jax.Array:
        del params, t
        hit_max = state.step >= self.max_steps
        positions = state.defenders.rtn[:, :3]
        min_dist = jnp.min(jnp.linalg.norm(positions, axis=-1))
        breached = min_dist < self.breach_distance_m
        return jnp.logical_or(hit_max, breached)
