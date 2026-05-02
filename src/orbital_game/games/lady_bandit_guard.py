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

from orbital_game.games.base import Game
from orbital_game.registry import GameKey, register_game


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
    n_guards: int = 1,
    n_bandits: int = 1,
    breach_distance_m: float = 10.0,
    max_horizon_s: float = 2000.0,
    dt: float = 10.0,
    seed: int = 0,
    **scenario_kwargs,
):
    """Builder for an LBG scenario.

    Wires `LadyBanditGuard(breach_distance_m=...)` as `cfg.game` and
    constructs a `MaxStepsOrBreach` termination using the same knob.

    Additional `scenario_kwargs` are passed through to `ScenarioConfig`
    (e.g. `reference_orbit`, `ic_sampler`, `guard_components`, etc.).
    Defaults assume a 1v1 RTN scenario with mass-tracked guard; for richer
    setups, the caller supplies the kinematic + IC parameters explicitly.
    """
    import jax.numpy as jnp

    from orbital_game.config import ScenarioConfig, VehicleParamsSpec
    from orbital_game.reference_orbit import ReferenceOrbitState
    from orbital_game.registry import StateComponentKey
    from orbital_game.sampling.mass import ConstantMass
    from orbital_game.sampling.side import RelativeEllipse
    from orbital_game.sampling.spec import ICSpec
    from orbital_game.termination.reference import MaxStepsOrBreach

    # Defaults match the bootstrap reference scenario for byte-identity:
    # 1km radial-ellipse co-orbit, guard at phase=0, bandit at phase=pi.
    defaults: dict = {
        "epoch_mjd_utc": 60067.0,
        "reference_orbit": ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        "guard_components": (StateComponentKey.RTN, StateComponentKey.MASS),
        "bandit_components": (StateComponentKey.RTN,),
        "guard_params": VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        "bandit_params": VehicleParamsSpec(dry_mass_kg=50.0, isp_s=200.0, max_thrust_n=2.0),
        "ic_sampler": ICSpec(
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
        ),
    }
    defaults.update(scenario_kwargs)

    cfg = ScenarioConfig(
        n_guards=n_guards,
        n_bandits=n_bandits,
        dt=dt,
        max_horizon_s=max_horizon_s,
        seed=seed,
        game=LadyBanditGuard(breach_distance_m=breach_distance_m),
        **defaults,
    )

    # Override the default termination_fn to use the LBG breach_distance_m.
    object.__setattr__(
        cfg,
        "termination_fn",
        MaxStepsOrBreach(max_steps=cfg.max_steps, breach_distance_m=breach_distance_m),
    )
    return cfg
