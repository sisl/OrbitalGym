"""ObservationFn protocol + scope enum.

Each side has its own ObservationFn. The protocol carries a `scope`
attribute that determines output shape:
  PER_VEHICLE: returns (N_side, obs_dim) — one observation per vehicle
  PER_SIDE:    returns (obs_dim,)        — one shared observation per side

The `side: Side` argument lets a single implementation handle both sides
when their sensor model is the same (just dispatching on `side`); for
genuinely asymmetric sensor suites, wire two different classes (one per
config field).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

import jax

from orbital_game.env.types import Side


class ObservationScope(StrEnum):
    PER_VEHICLE = "per_vehicle"
    PER_SIDE = "per_side"


class ObservationFn(Protocol):
    """Structural protocol for an observation function.

    Concrete implementations declare a `scope` class/instance attribute.
    """

    scope: ObservationScope

    def __call__(
        self,
        env_state: Any,
        side: Side,
        params: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> jax.Array: ...
