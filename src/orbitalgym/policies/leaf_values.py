"""Heuristic leaf values for MCTS on LBG, expressed in distance units."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def _positions(side_state: Any) -> jax.Array:
    if hasattr(side_state, "rtn"):
        return side_state.rtn[:, :3]
    return side_state.rt[:, :2]


def _distances(adapter: Any, s_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
    state = adapter.unpack(s_flat)
    guards = _positions(state.guards)
    bandits = _positions(state.bandits)
    d_gb = jnp.min(jnp.linalg.norm(guards[:, None, :] - bandits[None, :, :], axis=-1))
    d_bl = jnp.min(jnp.linalg.norm(bandits, axis=-1))
    return d_gb, d_bl


def guard_leaf_value(adapter: Any, scale_m: float = 300.0) -> Callable[[jax.Array], jax.Array]:
    """Higher when the nearest bandit is far from the lady and close to a guard."""

    def value(s_flat: jax.Array) -> jax.Array:
        d_gb, d_bl = _distances(adapter, s_flat)
        return ((d_bl - d_gb) / scale_m).astype(s_flat.dtype)

    return value


def bandit_leaf_value(adapter: Any, scale_m: float = 300.0) -> Callable[[jax.Array], jax.Array]:
    """Higher when the nearest bandit is close to the lady."""

    def value(s_flat: jax.Array) -> jax.Array:
        _, d_bl = _distances(adapter, s_flat)
        return (-d_bl / scale_m).astype(s_flat.dtype)

    return value
