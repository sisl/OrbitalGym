"""Per-side IC samplers — physical-orbit-respecting only.

Two implementations ship:
  - RelativeKeplerian: sample Keplerian element differences around HVA.
  - RelativeEllipse: sample NSROE with ellipse extents in meters (Task 8).

Both produce bounded relative orbits. Independent Gaussian/uniform sampling
on raw RTN state is intentionally not provided — it produces drift orbits
that violate the bounded-relative-orbit condition.

All samplers accept an optional mass_sampler that's consulted only if Mass
appears in the components list at __call__ time. If Mass is in components
and mass_sampler is None, ValueError is raised.

Convention: KOE vector layout follows astrojax = [a, e, i, RAAN, omega, M]
where M is mean anomaly. Singular at e=0, i=0; HVA should have small but
nonzero eccentricity / inclination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from astrojax import state_eci_to_koe, state_eci_to_rtn, state_koe_to_eci

from orbital_game.hva import mean_motion as _hva_mean_motion
from orbital_game.registry import (
    SideSamplerKey,
    StateComponentKey,
    register,
)
from orbital_game.sampling.mass import ConstantMass, UniformMass
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
    StateComponentKey.RT: RTState,
    StateComponentKey.RTN: RTNState,
    StateComponentKey.MASS: Mass,
    StateComponentKey.POWER: Power,
    StateComponentKey.ATTITUDE: Attitude,
    StateComponentKey.BODY_RATES: BodyRates,
}


def _build_side_class(components, n_vehicles, class_name):
    """Build the per-side flax dataclass type for the requested components."""
    comps = [_COMP_LOOKUP[c] for c in components]
    return build_state_class(comps, n_vehicles=n_vehicles, class_name=class_name)


def _maybe_sample_mass(mass_sampler, components, n_vehicles, key):
    """Return the propellant_mass array if Mass is in components, else None.

    Raises ValueError if Mass is requested but no mass_sampler was supplied.
    """
    if StateComponentKey.MASS not in components:
        return None
    if mass_sampler is None:
        raise ValueError(
            "Mass component is present but no mass_sampler was supplied. "
            "Use ConstantMass(...) or UniformMass(...) on the per-side sampler."
        )
    return mass_sampler(n_vehicles=n_vehicles, key=key)


def _broadcast_to_n(x, n):
    """Broadcast scalar/0-D to (n,); pass (n,) through; reject mismatched shapes."""
    a = jnp.asarray(x)
    if a.ndim == 0:
        return jnp.broadcast_to(a, (n,))
    if a.ndim == 1 and a.shape[0] == n:
        return a
    raise ValueError(f"Expected scalar or shape ({n},), got shape {a.shape}")


@register(SideSamplerKey.RELATIVE_KEPLERIAN)
@dataclass(frozen=True)
class RelativeKeplerian:
    """Sample Keplerian element differences around HVA's osculating elements.

    Each delta (delta_a, delta_e, delta_i, delta_RAAN, delta_omega, delta_M)
    is Gaussian: mean_* (default 0) and sigma_* (default 0). Scalars broadcast
    to all N vehicles; (N,) arrays specify per-vehicle parameters.
    M = mean anomaly (astrojax convention).

    Singular at HVA e=0, i=0 (RAAN/omega/M ill-defined). Use a slightly
    inclined or eccentric HVA for clean sampling.

    Precision note: this sampler relies on astrojax for KOE<->ECI conversions.
    The package configures astrojax for float64 at import time
    (`orbital_game/__init__.py`), giving sub-mm round-trip residuals on
    Earth-orbit scenarios.
    """

    mean_delta_sma_m: jnp.ndarray | float = 0.0
    sigma_delta_sma_m: jnp.ndarray | float = 0.0
    mean_delta_ecc: jnp.ndarray | float = 0.0
    sigma_delta_ecc: jnp.ndarray | float = 0.0
    mean_delta_inc_rad: jnp.ndarray | float = 0.0
    sigma_delta_inc_rad: jnp.ndarray | float = 0.0
    mean_delta_raan_rad: jnp.ndarray | float = 0.0
    sigma_delta_raan_rad: jnp.ndarray | float = 0.0
    mean_delta_argp_rad: jnp.ndarray | float = 0.0
    sigma_delta_argp_rad: jnp.ndarray | float = 0.0
    mean_delta_mean_anomaly_rad: jnp.ndarray | float = 0.0
    sigma_delta_mean_anomaly_rad: jnp.ndarray | float = 0.0
    mass_sampler: ConstantMass | UniformMass | None = None

    def __call__(
        self,
        config,
        key,
        *,
        n_vehicles,
        components,
        class_name,
    ) -> Any:
        cls = _build_side_class(components, n_vehicles, class_name)
        k_sample, k_mass = jax.random.split(key, 2)

        # 1. HVA osculating elements (KOE = [a, e, i, RAAN, omega, M])
        hva_state_eci = jnp.concatenate(
            [config.hva.position_eci, config.hva.velocity_eci]
        )
        hva_koe = state_eci_to_koe(hva_state_eci)

        # 2. Sample per-vehicle element-difference vectors
        means = jnp.stack(
            [
                _broadcast_to_n(self.mean_delta_sma_m, n_vehicles),
                _broadcast_to_n(self.mean_delta_ecc, n_vehicles),
                _broadcast_to_n(self.mean_delta_inc_rad, n_vehicles),
                _broadcast_to_n(self.mean_delta_raan_rad, n_vehicles),
                _broadcast_to_n(self.mean_delta_argp_rad, n_vehicles),
                _broadcast_to_n(self.mean_delta_mean_anomaly_rad, n_vehicles),
            ],
            axis=-1,
        )  # (n_vehicles, 6)
        sigmas = jnp.stack(
            [
                _broadcast_to_n(self.sigma_delta_sma_m, n_vehicles),
                _broadcast_to_n(self.sigma_delta_ecc, n_vehicles),
                _broadcast_to_n(self.sigma_delta_inc_rad, n_vehicles),
                _broadcast_to_n(self.sigma_delta_raan_rad, n_vehicles),
                _broadcast_to_n(self.sigma_delta_argp_rad, n_vehicles),
                _broadcast_to_n(self.sigma_delta_mean_anomaly_rad, n_vehicles),
            ],
            axis=-1,
        )

        noise = jax.random.normal(k_sample, (n_vehicles, 6))
        deltas = means + sigmas * noise

        # 3. Per-vehicle absolute KOE
        per_vehicle_koe = hva_koe[None, :] + deltas

        # 4. Convert each vehicle's KOE -> ECI -> HVA-RTN
        def _to_rtn(koe):
            state_eci = state_koe_to_eci(koe)
            return state_eci_to_rtn(hva_state_eci, state_eci)

        rtn_states = jax.vmap(_to_rtn)(per_vehicle_koe)

        # 5. RT scenarios drop cross-track + cross-track velocity
        if StateComponentKey.RT in components:
            rtn_states = jnp.stack(
                [rtn_states[:, 0], rtn_states[:, 1], rtn_states[:, 3], rtn_states[:, 4]],
                axis=-1,
            )

        # 6. Build the side state pytree
        kwargs = {}
        if StateComponentKey.RTN in components:
            kwargs["rtn"] = rtn_states
        elif StateComponentKey.RT in components:
            kwargs["rt"] = rtn_states

        propellant = _maybe_sample_mass(self.mass_sampler, components, n_vehicles, k_mass)
        if propellant is not None:
            kwargs["propellant_mass"] = propellant

        zeroed = cls.zeros(n_vehicles)
        return zeroed.replace(**kwargs)


@register(SideSamplerKey.RELATIVE_ELLIPSE)
@dataclass(frozen=True)
class RelativeEllipse:
    """Sample bounded relative orbits via NSROE with ellipse extents in meters.

    The non-singular relative orbital elements (Schaub & Junkins; Lyu & D'Amico)
    map to physical extents as:
        a * |delta_e| = radial-ellipse semi-major axis (m)
        a * |delta_i| = cross-track amplitude (m)
        a * delta_lambda = mean along-track separation (m)
        a * delta_a = secular drift coefficient (1.5*n*delta_a per radian)

    radial_ellipse_m / cross_track_m: scalar broadcasts; (N,) per-vehicle.
    along_track_offset_m: same.
    along_track_drift_m_per_orbit: optional; default 0 -> bounded.
    sigma_*: optional Gaussian noise added to the corresponding extent.
    phase_rad: per-vehicle initial phase along the relative ellipse.
        None -> uniform random per vehicle.
        scalar / (N,) -> deterministic per vehicle.
    """

    radial_ellipse_m: jnp.ndarray | float = 0.0
    cross_track_m: jnp.ndarray | float = 0.0
    along_track_offset_m: jnp.ndarray | float = 0.0
    along_track_drift_m_per_orbit: jnp.ndarray | float = 0.0
    sigma_radial_ellipse_m: jnp.ndarray | float = 0.0
    sigma_cross_track_m: jnp.ndarray | float = 0.0
    sigma_along_track_offset_m: jnp.ndarray | float = 0.0
    phase_rad: jnp.ndarray | float | None = None
    mass_sampler: ConstantMass | UniformMass | None = None

    def __call__(
        self,
        config,
        key,
        *,
        n_vehicles,
        components,
        class_name,
    ) -> Any:
        cls = _build_side_class(components, n_vehicles, class_name)
        k_extents, k_phase, k_mass = jax.random.split(key, 3)

        # Use the shared HVA mean-motion helper so sampler and dynamics agree
        # exactly on `n` — required for the IC to satisfy the bounded-orbit
        # condition under HCW.
        n_motion = _hva_mean_motion(config.hva)

        # Per-vehicle extents
        radial_e = _broadcast_to_n(self.radial_ellipse_m, n_vehicles)
        cross_e = _broadcast_to_n(self.cross_track_m, n_vehicles)
        along_off = _broadcast_to_n(self.along_track_offset_m, n_vehicles)
        drift = _broadcast_to_n(self.along_track_drift_m_per_orbit, n_vehicles)

        s_radial = _broadcast_to_n(self.sigma_radial_ellipse_m, n_vehicles)
        s_cross = _broadcast_to_n(self.sigma_cross_track_m, n_vehicles)
        s_along = _broadcast_to_n(self.sigma_along_track_offset_m, n_vehicles)

        k1, k2, k3 = jax.random.split(k_extents, 3)
        radial_e = radial_e + s_radial * jax.random.normal(k1, (n_vehicles,))
        cross_e = cross_e + s_cross * jax.random.normal(k2, (n_vehicles,))
        along_off = along_off + s_along * jax.random.normal(k3, (n_vehicles,))

        # Phase
        if self.phase_rad is None:
            phase = jax.random.uniform(k_phase, (n_vehicles,), minval=0.0, maxval=2 * jnp.pi)
        else:
            phase = _broadcast_to_n(self.phase_rad, n_vehicles)

        # NSROE -> RTN closed form, chief in circular orbit, t=0:
        #   r(t)    = -radial_e * cos(phase) + delta_a_coef
        #   t(t)    =  2 * radial_e * sin(phase) + along_off
        #   n_(t)   =  cross_e * sin(phase)
        #   rdot    =  n * radial_e * sin(phase)
        #   tdot    =  2 * n * radial_e * cos(phase) - 1.5 * n * delta_a_coef
        #   n_dot   =  n * cross_e * cos(phase)
        delta_a_coef = drift / (2.0 * jnp.pi)

        cp = jnp.cos(phase)
        sp = jnp.sin(phase)

        r_pos = -radial_e * cp + delta_a_coef
        t_pos = 2.0 * radial_e * sp + along_off
        n_pos = cross_e * sp
        r_vel = n_motion * radial_e * sp
        t_vel = 2.0 * n_motion * radial_e * cp - 1.5 * n_motion * delta_a_coef
        n_vel = n_motion * cross_e * cp

        rtn_states = jnp.stack([r_pos, t_pos, n_pos, r_vel, t_vel, n_vel], axis=-1)

        if StateComponentKey.RT in components:
            rtn_states = jnp.stack([r_pos, t_pos, r_vel, t_vel], axis=-1)

        kwargs = {}
        if StateComponentKey.RTN in components:
            kwargs["rtn"] = rtn_states
        elif StateComponentKey.RT in components:
            kwargs["rt"] = rtn_states

        propellant = _maybe_sample_mass(self.mass_sampler, components, n_vehicles, k_mass)
        if propellant is not None:
            kwargs["propellant_mass"] = propellant

        zeroed = cls.zeros(n_vehicles)
        return zeroed.replace(**kwargs)
