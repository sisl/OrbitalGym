"""ObservationFn protocol.

Each side has its own ObservationFn. The protocol's __call__ returns a tuple
of `Observation` channels — one per sensor modality. Belief updaters consume
the tuple and fold each channel's correction sequentially.

The `side: Side` argument lets a single implementation handle both sides
when their sensor model is the same (just dispatching on `side`); for
genuinely asymmetric sensor suites, wire two different classes (one per
config field).
"""

from __future__ import annotations

from typing import Any, Protocol

import jax

from orbitalgym.env.types import Side
from orbitalgym.observations.types import Observation


class ObservationFn(Protocol):
    """Structural protocol for an observation function.

    Returns a tuple of `Observation` channels. Most implementations return one
    channel; composite functions return more.

    `actions` is the full `Actions` pytree (both sides). When the observation
    fn is called pre-step (env.reset, single_agent's pre-decision opp obs),
    `actions` is the identity Command for both sides — no information leakage
    from a hypothetical step. When called post-step (env.step), `actions` is
    the action that produced `next_state`.
    """

    def __call__(
        self,
        env_state: Any,
        actions: Any,  # Actions — kept Any to avoid circular import on env.types
        side: Side,
        params: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Observation, ...]: ...
