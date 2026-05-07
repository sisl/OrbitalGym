"""Initial-condition sampling: ICSpec, per-side samplers, mass sub-samplers, validators."""

from orbital_game.sampling.attitude import (
    FixedAttitude,
    IdentityAttitude,
    UniformAttitude,
    UniformAttitudeAndRates,
    UniformBodyRates,
)
from orbital_game.sampling.mass import ConstantMass, UniformMass
from orbital_game.sampling.side import RelativeEllipse, RelativeKeplerian
from orbital_game.sampling.spec import ICSpec, SideSampler, Validator
from orbital_game.sampling.validators import (
    MaxRange,
    MinSeparation,
    SeparationScope,
    SideSelector,
)

__all__ = [
    "ConstantMass",
    "FixedAttitude",
    "ICSpec",
    "IdentityAttitude",
    "MaxRange",
    "MinSeparation",
    "RelativeEllipse",
    "RelativeKeplerian",
    "SeparationScope",
    "SideSampler",
    "SideSelector",
    "UniformAttitude",
    "UniformAttitudeAndRates",
    "UniformBodyRates",
    "UniformMass",
    "Validator",
]
