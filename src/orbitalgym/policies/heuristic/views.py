"""Agent-view readers shared by the heuristic policies."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

IN_PLANE_RTN = jnp.array([0, 1, 3, 4])


def mean_from_view(agent_view: Any, n_vehicles: int, n_opponents: int, state_dim: int) -> jax.Array:
    """The ``(N_obs, N_total, d)`` state block of a belief view or a flat observation.

    A belief exposes it as ``.mean``. A flat observation carries the same
    numbers in row-major order; when it holds several equal-sized full-state
    channels (as :class:`~orbitalgym.observations.composite.CompositeObservation`
    produces) the last channel is used. Any other size is rejected, because a
    position-only channel cannot supply the velocities these policies regulate.
    """
    if not isinstance(agent_view, jax.Array):
        return agent_view.mean
    n_total = n_vehicles + n_opponents
    expected_size = n_vehicles * n_total * state_dim
    size = agent_view.size
    if size == expected_size:
        flat = agent_view
    elif size % expected_size == 0:
        flat = agent_view.reshape((-1, expected_size))[-1]
    else:
        raise ValueError(
            f"Flat observation size {size} is not a multiple of expected "
            f"{expected_size} ({n_vehicles} vehicles × {n_total} entities × "
            f"{state_dim} dims). Flat-observation path requires one or more "
            f"equal-sized full-state channels; position-only channels must use a "
            f"belief view."
        )
    return flat.reshape((n_vehicles, n_total, state_dim))
