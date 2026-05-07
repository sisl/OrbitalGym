"""Observations package — observation function protocol and built-in implementations."""

from orbital_game.observations.base import ObservationFn
from orbital_game.observations.composite import CompositeObservation
from orbital_game.observations.conical import ConicalObservation  # noqa: F401
from orbital_game.observations.onboard_gps import OnboardGPSObservation
from orbital_game.observations.range_limited import RangeLimitedObservation
from orbital_game.observations.reference import FullObservation
from orbital_game.observations.types import Observation, flatten_observations

__all__ = [
    "CompositeObservation",
    "ConicalObservation",
    "FullObservation",
    "Observation",
    "ObservationFn",
    "OnboardGPSObservation",
    "RangeLimitedObservation",
    "flatten_observations",
]
