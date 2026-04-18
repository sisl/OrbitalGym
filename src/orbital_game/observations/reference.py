"""Reference observation: full flat state (no masking, no noise)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orbital_game.registry import ObservationFnKey, register


@register(ObservationFnKey.FULL)
@dataclass(frozen=True)
class FullObservation:
    """Return StateLayout.flatten(env_state.defenders, env_state.intruders).

    This is the no-partial-observability baseline. Real scenarios will swap this
    out for a function that masks/noises the intruder portion.
    """

    layout: Any  # StateLayout

    def __call__(self, env_state, params, key, t, for_side):
        del params, key, t, for_side
        return self.layout.flatten(env_state.defenders, env_state.intruders)
