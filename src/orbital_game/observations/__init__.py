"""Observations package — observation function protocol and built-in implementations."""

from orbital_game.observations.base import ObservationFn, ObservationScope
from orbital_game.observations.types import Observation, flatten_observations

__all__ = [
    "Observation",
    "ObservationFn",
    "ObservationScope",
    "flatten_observations",
]
