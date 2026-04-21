"""Reference initial-condition sampler: Gaussian noise around nominal states.

Works with either dynamics choice — the nominal arrays are `(n, 6)` for RTN or
`(n, 4)` for RT. Position and velocity components are split evenly (half of the
trailing axis each), so noise dimensions follow the state dimensionality.
"""

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
    """Seed-deterministic Gaussian perturbation around nominal states.

    Nominal state arrays have shape `(n, D)` where D is 6 for RTN scenarios or
    4 for RT scenarios. Position and velocity are split as `D/2` each. Variances
    sigma_pos / sigma_vel apply to position / velocity components respectively.
    """

    nominal_defender_state: jax.Array  # (n_defenders, 6) for RTN or (n_defenders, 4) for RT
    nominal_intruder_state: jax.Array  # (n_intruders, 6) for RTN or (n_intruders, 4) for RT
    sigma_pos: float
    sigma_vel: float

    def __call__(self, config, key):
        k_d, k_i = jax.random.split(key, 2)
        def_cls = _build_state(config.defender_components, config.n_defenders, "DefenderState")
        int_cls = _build_state(config.intruder_components, config.n_intruders, "IntruderState")

        def _sample(nominal, k):
            pos_dim = nominal.shape[-1] // 2
            k_pos, k_vel = jax.random.split(k, 2)
            noise_pos = jax.random.normal(k_pos, nominal[:, :pos_dim].shape) * self.sigma_pos
            noise_vel = jax.random.normal(k_vel, nominal[:, pos_dim:].shape) * self.sigma_vel
            return jnp.concatenate(
                [nominal[:, :pos_dim] + noise_pos, nominal[:, pos_dim:] + noise_vel], axis=-1
            )

        def_vec = _sample(self.nominal_defender_state, k_d)
        int_vec = _sample(self.nominal_intruder_state, k_i)

        # Construct zero-initialized states, then overwrite the dynamics leaf
        # (rtn or rt depending on which component is active).
        def_state = def_cls.zeros(config.n_defenders)
        int_state = int_cls.zeros(config.n_intruders)
        def_state = _replace_dynamics_state(def_state, def_vec)
        int_state = _replace_dynamics_state(int_state, int_vec)

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


def _replace_dynamics_state(state, new_vec):
    """Set the dynamics-state leaf (rtn or rt) via hasattr discrimination.

    Matches env/environment.py's _set_dynamics_state pattern, scoped here to
    avoid importing from env into sampling.
    """
    if hasattr(state, "rtn"):
        return state.replace(rtn=new_vec)
    if hasattr(state, "rt"):
        return state.replace(rt=new_vec)
    raise AttributeError("State has no dynamics component (RTState or RTNState)")
