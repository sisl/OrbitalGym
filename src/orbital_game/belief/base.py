"""Belief / BeliefInitializer / BeliefUpdater protocols — side-aware.

A `Belief` is any representation of the agent's uncertainty over the
environment state that exposes a `mean: jax.Array` state estimate. Concrete
beliefs (`KFBelief`, `EKFBelief`) satisfy this structurally — particle-based
beliefs added later expose `mean` as a `@property` computing the weighted
mean. Policies that need a single point estimate (e.g. `MCTSPolicy`'s
search root) read `belief.mean`; richer beliefs grow more methods over
time (e.g. `.sample(key, n)` for POMCP-style particle MCTS).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax

from orbital_game.env.types import Side


@runtime_checkable
class Belief(Protocol):
    """Structural protocol for any belief representation.

    Implementations must expose a ``mean: jax.Array`` attribute holding
    the per-pair state estimate. Shape conventions follow the concrete
    belief classes (``(N_obs, N_total, d)`` for KF/EKF).
    """

    mean: jax.Array


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
