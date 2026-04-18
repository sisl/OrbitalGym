"""IntruderPolicy protocol — structural contract for intruder behavior."""

from __future__ import annotations

from typing import Any, Protocol

import jax


class IntruderPolicy(Protocol):
    """Structural protocol for an intruder policy.

    Implementations return a command array shaped (n_intruders, action_dim).
    `action_dim` matches the dynamics choice (2 for HCW_RT, 3 for HCW_RTN).
    """

    def __call__(self, obs: Any, key: jax.Array, t: jax.Array) -> jax.Array: ...
