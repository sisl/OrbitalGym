"""RewardFn protocol + scope enum."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

import jax

from orbitalgym.env.types import Actions, Side


class RewardScope(StrEnum):
    PER_VEHICLE = "per_vehicle"
    PER_SIDE = "per_side"


class RewardFn(Protocol):
    """Structural protocol for a per-step reward function.

    Called as reward_fn(prev_state, action, next_state, side, params, t).
    `action` is a full Actions container (both sides) so multi-side
    rewards (e.g. zero-sum penalties using both sides' actions) can be
    expressed naturally. Returns shape per `scope`:
      PER_VEHICLE: (N_side,)
      PER_SIDE:    () (scalar)
    """

    scope: RewardScope

    def __call__(
        self,
        prev_state: Any,
        action: Actions,
        next_state: Any,
        side: Side,
        params: Any,
        t: jax.Array,
    ) -> jax.Array: ...
