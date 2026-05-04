"""CompositeActionPolicy — sum of base and offset policy actions.

Useful for "scripted base + learned residual" architectures where the
learned head provides corrections on top of a known-reasonable scripted
controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from orbital_game.policies.base import Policy


@dataclass(frozen=True)
class CompositeActionPolicy:
    """Sums actions from `base` and `offset`. Both must produce same-shape actions."""

    base: Policy
    offset: Policy
    n_vehicles: int = 0
    action_dim: int = 0

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, Any]:
        k1, k2 = jax.random.split(key)
        base_action, _ = self.base(None, obs, k1, t)
        offset_action, _ = self.offset(None, obs, k2, t)
        return base_action + offset_action, policy_state
