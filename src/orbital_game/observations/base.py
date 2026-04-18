"""ObservationFn protocol — structural contract for observation functions."""

from __future__ import annotations

from typing import Any, Literal, Protocol

import jax


class ObservationFn(Protocol):
    """Structural protocol for an observation function.

    Receives the full ground-truth env_state and returns what the side under
    `for_side` is allowed to see. Reference implementation returns the full
    flat state; realistic implementations mask/noise the intruder portion.
    """

    def __call__(
        self,
        env_state: Any,
        params: Any,
        key: jax.Array,
        t: jax.Array,
        for_side: Literal["defender", "intruder"],
    ) -> Any: ...
