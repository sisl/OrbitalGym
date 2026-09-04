"""TerminationFn protocol — episode-level."""

from __future__ import annotations

from typing import Any, Protocol

import jax


class TerminationFn(Protocol):
    """Episode-level termination — returns scalar bool.

    Per-side `done` in SideOutput is this scalar broadcast (kept for shape
    uniformity). Per-side termination (one side gives up, the other does
    not) is a Phase-N concern; not needed by the four bootstrap games.
    """

    def __call__(
        self,
        prev_state: Any,
        state: Any,
        params: Any,
        t: jax.Array,
    ) -> jax.Array: ...
