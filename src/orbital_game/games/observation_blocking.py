"""Observation-Blocking game.

Bandit blocks guard's view of an Earth surface target — but only when the
target is observable (guard above min_elevation_deg from the target).
Reward shape: Gaussian on collinearity angle, gated on (a) bandit farther
from target than guard, and (b) target visibility from guard.

Reward (PER_SIDE, zero-sum):
    bandit_reward = exp(-angle²/2σ²)  if visible AND ordering OK, else 0
    guard_reward  = -bandit_reward
"""

from __future__ import annotations

from dataclasses import dataclass, field

import astrojax
import jax
import jax.numpy as jnp

from orbital_game.games._frames import vehicle_eci_position
from orbital_game.games.base import Game
from orbital_game.registry import GameKey, RewardFnKey, register, register_game


def _epoch_from_mjd(mjd_float):
    """Construct an astrojax.Epoch from a Modified Julian Date (float).

    Delegates to the same helper in sun_blocking — see that module for the
    rationale (Epoch.__init__ only accepts (year, month, day, ...) or string).
    """
    from orbital_game.games.sun_blocking import _epoch_from_mjd as _sb_epoch_from_mjd

    return _sb_epoch_from_mjd(mjd_float)


@register_game(GameKey.OBSERVATION_BLOCKING)
@dataclass(frozen=True)
class ObservationBlocking(Game):
    """Bandit blocks guard's view of an Earth surface target.

    Knobs:
        target_lat_deg, target_lon_deg, target_alt_m: WGS84 geodetic
            coordinates of the surface target.
        min_elevation_deg: target visibility threshold (degrees above
            horizon). Reward is gated to 0 when the target is below this
            elevation as seen from the guard.
        angle_sigma_deg: collinearity Gaussian width (degrees).

    target_ecef_m is precomputed in __post_init__ (one-time geodetic→ECEF
    conversion) and cached on the instance for use in the reward fn.

    Note: astrojax.position_geodetic_to_ecef takes x_geod=[lon, lat, alt]
    (longitude first) with radians by default.
    """

    target_lat_deg: float = 37.4
    target_lon_deg: float = -122.2
    target_alt_m: float = 0.0
    min_elevation_deg: float = 5.0
    angle_sigma_deg: float = 5.0
    target_ecef_m: jax.Array = field(init=False)

    def __post_init__(self):
        # position_geodetic_to_ecef expects [lon, lat, alt] (lon first, radians).
        ecef = astrojax.position_geodetic_to_ecef(
            jnp.array(
                [
                    jnp.deg2rad(self.target_lon_deg),
                    jnp.deg2rad(self.target_lat_deg),
                    float(self.target_alt_m),
                ]
            )
        )
        object.__setattr__(self, "target_ecef_m", ecef)


@register(RewardFnKey.OBSERVATION_BLOCKING)
@dataclass(frozen=True)
class ObservationBlockingReward:
    """Earth-target collinearity reward, gated on visibility.

    Requires cfg.game: ObservationBlocking. Uses astrojax to convert
    target ECEF → ECI per timestep (via GMST rotation) and computes
    elevation from the guard using the manual dot-product formula.

    rotation_ecef_to_eci requires (EOPData, Epoch); we use zero_eop() for a
    simple GMST-only approximation (no EOP corrections), which is sufficient
    for game-play purposes.
    """

    # scope is a string to avoid circular import on RewardScope at module load.
    scope: str = "per_side"

    def __call__(self, prev_state, action, next_state, side, params, t):
        from orbital_game.env.types import Side

        del prev_state, action, t
        if not isinstance(params.game, ObservationBlocking):
            raise TypeError(
                f"ObservationBlockingReward requires cfg.game: ObservationBlocking; "
                f"got {type(params.game).__name__}"
            )

        guard_v0 = jax.tree_util.tree_map(lambda x: x[0], next_state.guards)
        bandit_v0 = jax.tree_util.tree_map(lambda x: x[0], next_state.bandits)

        guard_eci = vehicle_eci_position(
            guard_v0, params.reference_orbit, params.epoch_mjd_utc, next_state.t
        )
        bandit_eci = vehicle_eci_position(
            bandit_v0, params.reference_orbit, params.epoch_mjd_utc, next_state.t
        )

        # Target ECEF → ECI at current epoch+t.
        # rotation_ecef_to_eci(eop, epoch) — use zero_eop() for a simple
        # GMST-based approximation without EOP corrections.
        epoch = _epoch_from_mjd(params.epoch_mjd_utc + float(next_state.t) / 86400.0)
        eop = astrojax.zero_eop()
        rot_ecef_to_eci = astrojax.rotation_ecef_to_eci(eop, epoch)
        target_eci = rot_ecef_to_eci @ params.game.target_ecef_m

        # Elevation from target to guard (manual, avoids needing ECEF positions
        # for the guard and is JIT-compatible).
        # "up" at the target is the unit vector from Earth center to target.
        up = target_eci / jnp.linalg.norm(target_eci)
        dir_to_guard = (guard_eci - target_eci) / jnp.linalg.norm(guard_eci - target_eci)
        sin_el = jnp.dot(up, dir_to_guard)
        elevation_rad = jnp.arcsin(jnp.clip(sin_el, -1.0, 1.0))
        elevation_deg = jnp.rad2deg(elevation_rad)
        visible = elevation_deg >= params.game.min_elevation_deg

        # Collinearity: angle between (target→guard) and (target→bandit).
        u_tg = guard_eci - target_eci
        u_tb = bandit_eci - target_eci
        u_tg = u_tg / (jnp.linalg.norm(u_tg) + 1e-12)
        u_tb = u_tb / (jnp.linalg.norm(u_tb) + 1e-12)
        cos_theta = jnp.clip(jnp.dot(u_tg, u_tb), -1.0, 1.0)
        angle_deg = jnp.rad2deg(jnp.arccos(cos_theta))

        # Ordering: bandit must be farther from target than guard to actually
        # occlude the line-of-sight from the target's perspective.
        target_to_guard_dist = jnp.linalg.norm(guard_eci - target_eci)
        target_to_bandit_dist = jnp.linalg.norm(bandit_eci - target_eci)
        in_front = target_to_bandit_dist > target_to_guard_dist

        bandit_r = jnp.where(
            jnp.logical_and(visible, in_front),
            jnp.exp(-(angle_deg**2) / (2.0 * params.game.angle_sigma_deg**2)),
            0.0,
        )
        return jnp.where(side == Side.BANDIT, bandit_r, -bandit_r)


def make_observation_blocking(
    *,
    target_lat_deg: float = 37.4,
    target_lon_deg: float = -122.2,
    target_alt_m: float = 0.0,
    min_elevation_deg: float = 5.0,
    angle_sigma_deg: float = 5.0,
    n_guards: int = 1,
    n_bandits: int = 1,
    max_horizon_s: float = 5400.0,
    dt: float = 10.0,
    seed: int = 0,
    **scenario_kwargs,
):
    """Builder for an OB scenario."""
    import jax.numpy as jnp

    from orbital_game.config import ScenarioConfig, VehicleParamsSpec
    from orbital_game.reference_orbit import ReferenceOrbitState
    from orbital_game.registry import StateComponentKey
    from orbital_game.sampling.side import RelativeEllipse
    from orbital_game.sampling.spec import ICSpec
    from orbital_game.termination.reference import MaxStepsOrBreach

    defaults: dict = {
        "epoch_mjd_utc": 60067.0,
        "reference_orbit": ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        "guard_components": (StateComponentKey.RTN,),
        "bandit_components": (StateComponentKey.RTN,),
        "guard_params": VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        "bandit_params": VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0),
        "ic_sampler": ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=500.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1500.0,
                cross_track_m=100.0,
                along_track_offset_m=300.0,
                phase_rad=jnp.pi / 6.0,
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
        game=ObservationBlocking(
            target_lat_deg=target_lat_deg,
            target_lon_deg=target_lon_deg,
            target_alt_m=target_alt_m,
            min_elevation_deg=min_elevation_deg,
            angle_sigma_deg=angle_sigma_deg,
        ),
        reward_fn=ObservationBlockingReward(),
        **defaults,
    )
    object.__setattr__(
        cfg,
        "termination_fn",
        MaxStepsOrBreach(max_steps=cfg.max_steps, breach_distance_m=0.0),
    )
    return cfg
