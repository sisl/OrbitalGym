"""ObservationFn protocol + scope enum.

Each side has its own ObservationFn. The protocol's __call__ returns a tuple
of `Observation` channels — one per sensor modality. Belief updaters consume
the tuple and fold each channel's correction sequentially.

The `side: Side` argument lets a single implementation handle both sides
when their sensor model is the same (just dispatching on `side`); for
genuinely asymmetric sensor suites, wire two different classes (one per
config field).

`ObservationScope` is retained for legacy reference but is no longer used by
the new multi-channel implementations.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

import jax

from orbital_game.env.types import Side
from orbital_game.observations.types import Observation


class ObservationScope(StrEnum):
    PER_VEHICLE = "per_vehicle"
    PER_SIDE = "per_side"


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
