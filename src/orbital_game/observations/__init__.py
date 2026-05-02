"""Observations package — observation function protocol and built-in implementations."""

from orbital_game.observations.base import ObservationFn, ObservationScope
from orbital_game.observations.onboard_gps import OnboardGPSObservation
from orbital_game.observations.range_limited import RangeLimitedObservation
from orbital_game.observations.reference import FullObservation
from orbital_game.observations.types import Observation, flatten_observations

__all__ = [
    "FullObservation",
    "Observation",
    "ObservationFn",
    "ObservationScope",
    "OnboardGPSObservation",
    "RangeLimitedObservation",
    "flatten_observations",
]
