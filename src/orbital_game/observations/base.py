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

from orbital_game.env.types import Side
from orbital_game.observations.types import Observation


class ObservationFn(Protocol):
    """Structural protocol for an observation function.

    Returns a tuple of `Observation` channels. Most implementations return one
    channel; composite functions return more.
    """

    def __call__(
        self,
        env_state: Any,
        side: Side,
        params: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Observation, ...]: ...
