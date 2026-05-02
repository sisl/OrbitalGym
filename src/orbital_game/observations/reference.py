"""Reference observation: full flat state (no masking, no noise)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orbital_game.observations.base import ObservationScope
from orbital_game.registry import ObservationFnKey, register


@register(ObservationFnKey.FULL)
@dataclass(frozen=True)
class FullObservation:
    """Return StateLayout.flatten(env_state.guards, env_state.bandits).

    Side-agnostic by construction — both sides see the same full flat state.
    Realistic asymmetric models would use two different classes per side.
    """

    layout: Any  # StateLayout
    scope: ObservationScope = ObservationScope.PER_SIDE

    def __call__(self, env_state, side, params, key, t):
        del side, params, key, t
        return self.layout.flatten(env_state.guards, env_state.bandits)
