"""Mass sub-samplers for IC initialization.

Each mass sampler is a frozen dataclass with __call__(n_vehicles, key) ->
(n_vehicles,) jax.Array of propellant_mass values. Per-side samplers consult
the mass sampler only when `Mass` appears in the components list.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.registry import MassSamplerKey, register


@register(MassSamplerKey.CONSTANT)
@dataclass(frozen=True)
class ConstantMass:
    """Same propellant mass for all N vehicles."""

    propellant_mass_kg: float

    def __call__(self, n_vehicles: int, key: jax.Array) -> jax.Array:
        del key
        return jnp.full((n_vehicles,), self.propellant_mass_kg)


@register(MassSamplerKey.UNIFORM)
@dataclass(frozen=True)
class UniformMass:
    """Independent uniform [low, high] propellant mass per vehicle."""

    low_kg: float
    high_kg: float

    def __post_init__(self) -> None:
        if self.low_kg > self.high_kg:
            raise ValueError(
                f"UniformMass requires low_kg <= high_kg, got {self.low_kg} and {self.high_kg}"
            )

    def __call__(self, n_vehicles: int, key: jax.Array) -> jax.Array:
        return jax.random.uniform(
            key, shape=(n_vehicles,), minval=self.low_kg, maxval=self.high_kg
        )
