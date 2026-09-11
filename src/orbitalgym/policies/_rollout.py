"""Compatibility helpers for planner transition models."""

from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.env.types import Side


def step_with_termination(
    adapter: Any, s: jax.Array, action: jax.Array, key: jax.Array, side: Side
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Read optional termination metadata; legacy models are continuing tasks."""
    if hasattr(adapter, "step_with_termination"):
        return adapter.step_with_termination(s, action, key, side)
    s_next, reward = adapter.step(s, action, key, side)
    return s_next, reward, jnp.asarray(False)
