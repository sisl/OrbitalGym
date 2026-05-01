"""Reference observation: full flat state (no masking, no noise)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orbital_game.registry import ObservationFnKey, register


@register(ObservationFnKey.FULL)
@dataclass(frozen=True)
class FullObservation:
    """Return StateLayout.flatten(env_state.guards, env_state.bandits).

    The no-partial-observability baseline. Same behavior regardless of which
    side invokes it; asymmetric observation models use separate
    `FullObservation`-shaped implementations per side.
    """

    layout: Any  # StateLayout

    def __call__(self, env_state, params, key, t):
        del params, key, t
        return self.layout.flatten(env_state.guards, env_state.bandits)
