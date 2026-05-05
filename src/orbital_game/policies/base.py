"""Policy protocol — role-agnostic.

A Policy is a pure callable mapping (policy_state, obs, key, t) to a tuple
(command, next_policy_state). Stateful by default; stateless policies pass
None for policy_state. The same concrete Policy class can be wired as the
guard policy, bandit policy, or learned controlled side — only the per-side
Command class and n_vehicles injection differ at env-build.

The returned `command` is a per-side Command pytree (built by
`actions.assemble.build_command_class`), not a raw `(N, action_dim)` array.
Each component contributes its own field(s) to the Command; ZeroControl
emits the identity (zeros) for every component.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax


class Policy(Protocol):
    """Structural protocol for a role-agnostic policy.

    Implementations are typically frozen dataclasses with a `command_cls`
    field (the per-side Command class) and an `n_vehicles` field
    (defaulting to None / 0) that the env populates via
    `dataclasses.replace` at construction. The call returns a Command
    pytree — never a raw array.
    """

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]: ...
