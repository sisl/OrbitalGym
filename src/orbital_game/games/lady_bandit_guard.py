"""Lady-Bandit-Guard game.

Guard protects the reference orbit (the 'Lady', currently virtual at the
reference) from the bandit. Reward = -sum(guard distances to reference
orbit origin). Termination on max_steps OR guard within breach_distance_m
of the origin.

This Phase-2 implementation reuses the bootstrap's DistanceToReferenceOrbit
reward and MaxStepsOrBreach termination unchanged — LBG is a typed label
that carries the breach_distance_m knob to the builder, which wires it
into the existing termination class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from orbital_game.games.base import Game
from orbital_game.registry import (
    ActuatorKey,
    DynamicsKey,
    GameKey,
    StateComponentKey,
    register_game,
)

if TYPE_CHECKING:
    from orbital_game.config import VehicleParamsSpec
    from orbital_game.reference_orbit import ReferenceOrbitState
    from orbital_game.sampling.spec import ICSpec


@register_game(GameKey.LADY_BANDIT_GUARD)
@dataclass(frozen=True)
class LadyBanditGuard(Game):
    """Guard protects the reference orbit from the bandit.

    Knobs:
        breach_distance_m: Termination triggers when any guard is closer
            than this distance to the reference-orbit origin (RTN frame).
    """

    breach_distance_m: float = 10.0


def make_lady_bandit_guard(
    *,
    # Game-specific knob
    breach_distance_m: float = 10.0,
    # Fleet sizing
    n_guards: int = 1,
    n_bandits: int = 1,
    # Time + RNG
    dt: float = 10.0,
    max_horizon_s: float = 2000.0,
    seed: int = 0,
    # Time anchor + reference frame
    epoch_mjd_utc: float = 60067.0,
    reference_orbit: ReferenceOrbitState | None = None,
    # Per-side state composition (must match dynamics frame)
    guard_components: tuple[StateComponentKey, ...] = (
        StateComponentKey.RTN,
        StateComponentKey.MASS,
    ),
    bandit_components: tuple[StateComponentKey, ...] = (StateComponentKey.RTN,),
    # Per-side vehicle params
    guard_params: VehicleParamsSpec | None = None,
    bandit_params: VehicleParamsSpec | None = None,
    # IC sampling
    ic_sampler: ICSpec | None = None,
    # Dynamics + actuators
    truth_dynamics: DynamicsKey = DynamicsKey.HCW_RTN,
    policy_dynamics: DynamicsKey = DynamicsKey.HCW_RTN,
    guard_actuator: ActuatorKey = ActuatorKey.IMPULSIVE,
    bandit_actuator: ActuatorKey = ActuatorKey.IMPULSIVE,
    # Observation fns (None → ScenarioConfig.__post_init__ supplies FullObservation)
    guard_observation_fn: Any = None,
    bandit_observation_fn: Any = None,
):
    """Builder for an LBG scenario.

    Wires `LadyBanditGuard(breach_distance_m=...)` as `cfg.game` and
    constructs a `MaxStepsOrBreach` termination using the same knob.

    Defaults assume a 1v1 RTN scenario with mass-tracked guard. To switch
    to 2D RT dynamics, pass matching `truth_dynamics`/`policy_dynamics`
    *and* corresponding `guard_components`/`bandit_components`; otherwise
    `ScenarioConfig.__post_init__` will reject the incoherent combo.
    """
    import jax.numpy as jnp

    from orbital_game.config import ScenarioConfig, VehicleParamsSpec
    from orbital_game.reference_orbit import ReferenceOrbitState
    from orbital_game.sampling.mass import ConstantMass
    from orbital_game.sampling.side import RelativeEllipse
    from orbital_game.sampling.spec import ICSpec
    from orbital_game.termination.reference import MaxStepsOrBreach

    if reference_orbit is None:
        reference_orbit = ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        )
    if guard_params is None:
        guard_params = VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0)
    if bandit_params is None:
        bandit_params = VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=2.0)
    if ic_sampler is None:
        # Bootstrap reference scenario: 1km radial-ellipse co-orbit, guard at
        # phase=0, bandit at phase=pi.
        ic_sampler = ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=jnp.pi,
                sigma_radial_ellipse_m=10.0,
            ),
            validators=(),
            max_attempts=100,
        )

    cfg = ScenarioConfig(
        n_guards=n_guards,
        n_bandits=n_bandits,
        epoch_mjd_utc=epoch_mjd_utc,
        reference_orbit=reference_orbit,
        guard_components=guard_components,
        bandit_components=bandit_components,
        guard_params=guard_params,
        bandit_params=bandit_params,
        ic_sampler=ic_sampler,
        dt=dt,
        max_horizon_s=max_horizon_s,
        seed=seed,
        truth_dynamics=truth_dynamics,
        policy_dynamics=policy_dynamics,
        guard_actuator=guard_actuator,
        bandit_actuator=bandit_actuator,
        guard_observation_fn=guard_observation_fn,
        bandit_observation_fn=bandit_observation_fn,
        game=LadyBanditGuard(breach_distance_m=breach_distance_m),
    )

    # Override the default termination_fn to use the LBG breach_distance_m.
    object.__setattr__(
        cfg,
        "termination_fn",
        MaxStepsOrBreach(max_steps=cfg.max_steps, breach_distance_m=breach_distance_m),
    )
    return cfg
