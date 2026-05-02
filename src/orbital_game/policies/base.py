"""Policy protocol — role-agnostic.

A Policy is a pure callable mapping (policy_state, obs, key, t) to a tuple
(action, next_policy_state). Stateful by default; stateless policies pass
None for policy_state. The same concrete Policy class can be wired as a
guard scripted-opponent, bandit scripted-opponent, or learned controlled
side — only the (n_vehicles, action_dim) injection differs at env-build.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax


class Policy(Protocol):
    """Structural protocol for a role-agnostic policy.

    Implementations are typically frozen dataclasses with `n_vehicles` and
    `action_dim` fields (defaulting to 0) that the env populates via
    `dataclasses.replace` at construction.
    """

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, Any]: ...
