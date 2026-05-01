"""Reference termination: episode ends on max_steps OR guard breach radius."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.registry import TerminationFnKey, register


@register(TerminationFnKey.MAX_STEPS_OR_BREACH)
@dataclass(frozen=True)
class MaxStepsOrBreach:
    """Terminate when step count hits max_steps OR when any guard is inside
    breach_distance_m of the reference orbit origin.

    Works with either dynamics choice: RTN state → 3D norm over (r, t, n);
    RT state → 2D norm over (r, t).
    """

    max_steps: int
    breach_distance_m: float

    def __call__(self, state, params, t) -> jax.Array:
        del params, t
        hit_max = state.step >= self.max_steps
        guards = state.guards
        # RTN state → 3D norm over (r, t, n); RT state → 2D norm over (r, t).
        positions = guards.rtn[:, :3] if hasattr(guards, "rtn") else guards.rt[:, :2]
        min_dist = jnp.min(jnp.linalg.norm(positions, axis=-1))
        breached = min_dist < self.breach_distance_m
        return jnp.logical_or(hit_max, breached)
