"""RewardFn protocol — structural contract for reward functions."""

from __future__ import annotations

from typing import Any, Protocol

import jax


class RewardFn(Protocol):
    """Structural protocol for a scalar reward function.

    Called as reward_fn(prev_state, action, next_state, params, t).
    Returns a scalar jax.Array; convention is larger = better.
    """

    def __call__(
        self,
        prev_state: Any,
        action: Any,
        next_state: Any,
        params: Any,
        t: jax.Array,
    ) -> jax.Array: ...
