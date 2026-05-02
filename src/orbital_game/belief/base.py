"""BeliefInitializer / BeliefUpdater protocols — side-aware."""

from __future__ import annotations

from typing import Any, Protocol

import jax

from orbital_game.env.types import Side


class BeliefInitializer(Protocol):
    """Initialize a belief state from the env's ground-truth state."""

    def __call__(
        self,
        env_state: Any,
        side: Side,
        key: jax.Array,
    ) -> Any: ...


class BeliefUpdater(Protocol):
    """Update a belief given a new observation."""

    def __call__(
        self,
        belief: Any,
        obs: jax.Array,
        action: jax.Array,
        side: Side,
        key: jax.Array,
    ) -> Any: ...
