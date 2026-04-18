"""Reference initial-condition sampler: Gaussian noise around nominal RTN states."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.registry import InitialConditionSamplerKey, register
from orbital_game.state.assemble import build_state_class
from orbital_game.state.components import (
    Attitude,
    BodyRates,
    Mass,
    Power,
    RTNState,
    RTState,
)

_COMP_LOOKUP = {
    "rt": RTState,
    "rtn": RTNState,
    "mass": Mass,
    "power": Power,
    "attitude": Attitude,
    "body_rates": BodyRates,
}


def _build_state(components_keys, n_vehicles: int, class_name: str):
    comps = [_COMP_LOOKUP[k.value] for k in components_keys]
    return build_state_class(comps, n_vehicles=n_vehicles, class_name=class_name)


@register(InitialConditionSamplerKey.GAUSSIAN_AROUND_NOMINAL)
@dataclass(frozen=True)
class GaussianAroundNominal:
    """Seed-deterministic Gaussian perturbation around nominal RTN states.

    Nominal RTN arrays are supplied at construction time (one (n, 6) matrix per side).
    Variances sigma_pos / sigma_vel apply to position / velocity components respectively.

    Assumes the scenario uses RTNState (populates the `rtn` field). If you add
    a 2D (RTState) scenario later, this sampler needs extending.
    """

    nominal_defender_rtn: jax.Array  # (n_defenders, 6)
    nominal_intruder_rtn: jax.Array  # (n_intruders, 6)
    sigma_pos: float
    sigma_vel: float

    def __call__(self, config, key):
        k_d, k_i = jax.random.split(key, 2)
        def_cls = _build_state(config.defender_components, config.n_defenders, "DefenderState")
        int_cls = _build_state(config.intruder_components, config.n_intruders, "IntruderState")

        def _sample(nominal, k):
            k_pos, k_vel = jax.random.split(k, 2)
            noise_pos = jax.random.normal(k_pos, nominal[:, :3].shape) * self.sigma_pos
            noise_vel = jax.random.normal(k_vel, nominal[:, 3:].shape) * self.sigma_vel
            return jnp.concatenate(
                [nominal[:, :3] + noise_pos, nominal[:, 3:] + noise_vel], axis=-1
            )

        # Construct states with rtn perturbed; other components (Mass, Power, ...) at zeros
        # except Mass.propellant_mass which is initialized from params.
        def_rtn = _sample(self.nominal_defender_rtn, k_d)
        int_rtn = _sample(self.nominal_intruder_rtn, k_i)

        def_state = def_cls.zeros(config.n_defenders).replace(rtn=def_rtn)
        int_state = int_cls.zeros(config.n_intruders).replace(rtn=int_rtn)

        # Seed propellant if Mass is in the defender components.
        if Mass in def_state._orbital_game_components:
            def_state = def_state.replace(
                propellant_mass=jnp.full(
                    (config.n_defenders,), config.defender_params.propellant_mass_kg
                )
            )
        if Mass in int_state._orbital_game_components:
            int_state = int_state.replace(
                propellant_mass=jnp.full(
                    (config.n_intruders,), config.intruder_params.propellant_mass_kg
                )
            )
        return def_state, int_state
