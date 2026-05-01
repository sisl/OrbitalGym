"""BanditPolicy protocol — structural contract for bandit behavior."""

from __future__ import annotations

from typing import Any, Protocol

import jax


class BanditPolicy(Protocol):
    """Structural protocol for a bandit policy.

    Implementations return a command array shaped (n_bandits, action_dim).
    `action_dim` matches the dynamics choice (2 for HCW_RT, 3 for HCW_RTN).
    """

    def __call__(self, obs: Any, key: jax.Array, t: jax.Array) -> jax.Array: ...
