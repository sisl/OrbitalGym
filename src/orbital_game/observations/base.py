"""ObservationFn protocol — structural contract for observation functions.

Guards and bandits get their own independent observation functions —
they are not constrained to share sensor suites, masking patterns, or noise
models. `OrbitalGameEnv` wires a `guard_observation_fn` and a
`bandit_observation_fn` separately, both satisfying this same protocol.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax


class ObservationFn(Protocol):
    """Structural protocol for an observation function.

    Receives the full ground-truth env_state and returns what the caller
    (guard or bandit — the env decides by which attribute it invokes)
    is allowed to see. Reference implementation returns the full flat state;
    realistic implementations mask, slice, or add noise per their sensor model.
    """

    def __call__(
        self,
        env_state: Any,
        params: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> Any: ...
