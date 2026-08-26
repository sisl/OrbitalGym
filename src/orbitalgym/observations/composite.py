"""CompositeObservation - concatenates channels from multiple ObservationFns.

Splits the input key per constituent so each one gets a fresh subkey for noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

from orbitalgym.observations.types import Observation
from orbitalgym.registry import ObservationFnKey, register


@register(ObservationFnKey.COMPOSITE)
@dataclass(frozen=True)
class CompositeObservation:
    """Multi-channel observation: tuple of constituent observation fns.

    Each constituent runs against the same env_state with a fresh PRNG subkey
    and contributes its channel(s) to the concatenated tuple.
    """

    constituents: tuple

    def __call__(self, env_state, actions, side, params, key, t) -> tuple[Observation, ...]:
        subkeys = jax.random.split(key, max(len(self.constituents), 1))
        result: list[Observation] = []
        for fn, k in zip(self.constituents, subkeys, strict=True):
            channels = fn(env_state, actions, side, params, k, t)
            result.extend(channels)
        return tuple(result)
