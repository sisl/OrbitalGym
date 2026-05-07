"""Lady-Bandit-Guard ring-intercept scenario builder.

Composes existing primitives to produce a scenario where:

- Lady is a virtual fixed point at the RTN origin (the reference orbit).
- Bandits sit on a 2:1 RT-plane natural-motion ring around the lady at uniform
  phases (deterministic, not random).
- Both sides use HCW_RT (2D in-plane) dynamics with mass tracking.
- Vehicle parameters reflect a Moog 58E143-style 3.6 N high-thrust cold-gas
  thruster (GN2, Isp ~65 s) on a small inspector class with ~3 kg propellant.
- Reward is `LbgZeroSumReward` (dense distance shaping + terminal events at
  catch_radius / breach_radius).

The bandit and guard policies are wired separately by the caller — see
`examples/policies/lqr_bandit.py` for a JAX-native LQR bandit and
`orbital_game.policies.mcts.MCTSPolicy` for the guard.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.games.lady_bandit_guard import LadyBanditGuard
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import DynamicsKey, StateComponentKey
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec

# ---- Cold-gas thruster preset --------------------------------------------------

# Moog 58E143-001 GN2 cold-gas thruster, "high-thrust" 3.6 N configuration.
# Isp ~65 s for cold-gas GN2; dry mass + tank ~12 kg for a small inspector.
COLD_GAS_GUARD = VehicleParamsSpec(
    dry_mass_kg=12.0,
    isp_s=65.0,
    max_thrust_n=3.6,
)
COLD_GAS_BANDIT = VehicleParamsSpec(
    dry_mass_kg=12.0,
    isp_s=65.0,
    max_thrust_n=3.6,
)
# Initial GN2 tank load (kg). With Isp=65s, this gives total Δv budget
# of ~Isp * g0 * ln((dry+prop)/dry) ≈ 65 * 9.81 * ln(15/12) ≈ 142 m/s.
PROPELLANT_KG = 3.0


# ---- Scenario builder ----------------------------------------------------------


@dataclass(frozen=True)
class RingInterceptParams:
    n_bandits: int = 4
    n_guards: int = 1
    ring_radius_m: float = 2000.0  # radial-ellipse semi-major axis -> 2km × 4km ring
    guard_phase_offset_rad: float = 0.0  # rotates the whole guard ring placement
    guard_ring_radius_m: float = 200.0  # guard starts close to lady, defending
    breach_radius_m: float = 5.0
    catch_radius_m: float = 50.0
    dt: float = 10.0
    max_horizon_s: float = 4000.0
    seed: int = 0


def build_scenario(params: RingInterceptParams | None = None) -> ScenarioConfig:
    """Construct a ring-intercept LBG ScenarioConfig.

    The guard starts on a smaller ring (radius `guard_ring_radius_m`) so it
    naturally orbits the lady — close enough to defend but with freedom to
    maneuver. Bandits start on the outer ring at uniform phases. Both sides
    scale to any N: vehicles are placed at uniform phases around their ring.
    """
    if params is None:
        params = RingInterceptParams()
    n_b = params.n_bandits
    n_g = params.n_guards

    # Reference orbit: 7000 km circular-ish. Slightly inclined / eccentric to
    # avoid sampler singularities, though RelativeEllipse uses NSROE so it's
    # robust at e=0, i=0 anyway.
    reference_orbit = ReferenceOrbitState(
        position_eci=jnp.array([7000e3, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
    )

    # Per-side phases: uniform around 2π, deterministic. Single-vehicle case
    # collapses linspace to [0.0], recovering the original 1-vehicle placement.
    bandit_phases = jnp.linspace(0.0, 2.0 * jnp.pi, n_b, endpoint=False)
    guard_phases = (
        jnp.linspace(0.0, 2.0 * jnp.pi, n_g, endpoint=False) + params.guard_phase_offset_rad
    )

    ic_sampler = ICSpec(
        guard_sampler=RelativeEllipse(
            radial_ellipse_m=params.guard_ring_radius_m,
            cross_track_m=0.0,
            along_track_offset_m=0.0,
            phase_rad=guard_phases,
            mass_sampler=ConstantMass(propellant_mass_kg=PROPELLANT_KG),
        ),
        bandit_sampler=RelativeEllipse(
            radial_ellipse_m=params.ring_radius_m,
            cross_track_m=0.0,
            along_track_offset_m=0.0,
            phase_rad=bandit_phases,
            mass_sampler=ConstantMass(propellant_mass_kg=PROPELLANT_KG),
        ),
        validators=(),
        max_attempts=10,
    )

    cfg = ScenarioConfig(
        n_guards=n_g,
        n_bandits=n_b,
        epoch_mjd_utc=60067.0,
        reference_orbit=reference_orbit,
        # RT-plane only, both sides track mass.
        guard_components=(StateComponentKey.RT, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RT, StateComponentKey.MASS),
        guard_params=COLD_GAS_GUARD,
        bandit_params=COLD_GAS_BANDIT,
        ic_sampler=ic_sampler,
        dt=params.dt,
        max_horizon_s=params.max_horizon_s,
        seed=params.seed,
        truth_dynamics=DynamicsKey.HCW_RT,
        policy_dynamics=DynamicsKey.HCW_RT,
        game=LadyBanditGuard(
            breach_radius_m=params.breach_radius_m,
            catch_radius_m=params.catch_radius_m,
        ),
    )
    return cfg
