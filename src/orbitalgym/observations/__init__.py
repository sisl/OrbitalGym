"""Observations package — observation function protocol and built-in implementations."""

from orbitalgym.observations.base import ObservationFn
from orbitalgym.observations.composite import CompositeObservation
from orbitalgym.observations.conical import ConicalObservation  # noqa: F401
from orbitalgym.observations.negative_info import (
    Hard,
    NegativeInfoMode,
    Off,
    Soft,
    softness_from_sigma,
)
from orbitalgym.observations.onboard_gps import OnboardGPSObservation
from orbitalgym.observations.range_limited import RangeLimitedObservation
from orbitalgym.observations.reference import FullObservation
from orbitalgym.observations.types import Observation, flatten_observations

__all__ = [
    "CompositeObservation",
    "ConicalObservation",
    "FullObservation",
    "Hard",
    "NegativeInfoMode",
    "Observation",
    "ObservationFn",
    "Off",
    "OnboardGPSObservation",
    "RangeLimitedObservation",
    "Soft",
    "flatten_observations",
    "softness_from_sigma",
]
