"""Impulsive actuator: command is a velocity impulse per vehicle per step."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.actuators.base import AppliedControl
from orbital_game.registry import ActuatorKey, register

G0 = 9.80665  # m/s^2 — standard gravity for Isp calcs


@register(ActuatorKey.IMPULSIVE)
@dataclass(frozen=True)
class ImpulsiveActuator:
    """Impulsive Δv actuator.

    track_mass = True  => propellant is depleted per the rocket equation:
                          Δm = (m_dry + m_prop) * (1 - exp(-|Δv| / (Isp·g0)))
    track_mass = False => Δm is zeros; "infinite fuel" mode.

    The flag is a Python bool — it resolves statically at trace time so there is
    no dynamic branching in the jit-compiled code path.
    """

    track_mass: bool = True

    def apply(
        self, command: jax.Array, state, params, dt: float
    ) -> tuple[AppliedControl, jax.Array]:
        dv_mag = jnp.linalg.norm(command, axis=-1)  # (n,)
        if self.track_mass:
            wet_mass = params.dry_mass_kg + state.propellant_mass
            delta_propellant = wet_mass * (1.0 - jnp.exp(-dv_mag / (params.isp_s * G0)))
        else:
            delta_propellant = jnp.zeros_like(dv_mag)
        return AppliedControl(dv=command), delta_propellant
