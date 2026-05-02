"""Sun-Blocking game.

Bandit is rewarded for occluding the guard's view of the Sun — i.e.,
forming a Sun→Guard→Bandit collinear line with the bandit on the
far-from-sun side of the guard. Reward shape is a Gaussian on the
collinearity angle, gated on the ordering condition.

Reward (PER_SIDE, zero-sum):
    bandit_reward = exp(-angle² / (2σ²))    if bandit farther-from-sun than guard, else 0
    guard_reward  = -bandit_reward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import astrojax
import jax
import jax.numpy as jnp

from orbital_game.games._frames import vehicle_eci_position
from orbital_game.games.base import Game
from orbital_game.registry import (
    ActuatorKey,
    DynamicsKey,
    GameKey,
    RewardFnKey,
    StateComponentKey,
    register,
    register_game,
)

if TYPE_CHECKING:
    from orbital_game.config import VehicleParamsSpec
    from orbital_game.reference_orbit import ReferenceOrbitState
    from orbital_game.sampling.spec import ICSpec


def _epoch_from_mjd(mjd_float):
    """Construct an astrojax.Epoch from a Modified Julian Date (float).

    Epoch.__init__ only accepts (year, month, day, ...) or string — not a
    plain MJD float. We compute the Julian Day number and intra-day seconds
    via the same split-representation that Epoch uses internally.
    """
    jd_offset = astrojax.JD_MJD_OFFSET  # 2400000.5
    jd_full = float(mjd_float) + float(jd_offset)
    import math

    jd_int = math.floor(jd_full)
    seconds = (jd_full - jd_int) * 86400.0
    return astrojax.Epoch._from_internal(
        jnp.int32(jd_int),
        jnp.float32(seconds),
        jnp.float32(0.0),
    )


@register_game(GameKey.SUN_BLOCKING)
@dataclass(frozen=True)
class SunBlocking(Game):
    """1v1 sun-blocking game.

    Knobs:
        angle_sigma_deg: collinearity-angle Gaussian width (degrees).
            Smaller σ → narrower reward peak around perfect collinearity.
    """

    angle_sigma_deg: float = 5.0


@register(RewardFnKey.SUN_BLOCKING)
@dataclass(frozen=True)
class SunBlockingReward:
    """Bandit-collinear-with-Sun-Guard reward.

    Requires cfg.game: SunBlocking and cfg.epoch_mjd_utc / cfg.reference_orbit.
    Computes vehicle ECI positions via the games/_frames helper, looks up
    Sun ECI via astrojax.sun_position, then evaluates a Gaussian on the
    collinearity angle gated on the ordering condition.
    """

    # scope is a string to avoid circular import on RewardScope at module load.
    scope: str = "per_side"

    def __call__(self, prev_state, action, next_state, side, params, t):
        from orbital_game.env.types import Side

        del prev_state, action, t
        if not isinstance(params.game, SunBlocking):
            raise TypeError(
                f"SunBlockingReward requires cfg.game: SunBlocking; "
                f"got {type(params.game).__name__}"
            )

        # Vehicle ECI positions (1v1 — first guard, first bandit).
        guard_v0 = jax.tree_util.tree_map(lambda x: x[0], next_state.guards)
        bandit_v0 = jax.tree_util.tree_map(lambda x: x[0], next_state.bandits)

        guard_eci = vehicle_eci_position(
            guard_v0, params.reference_orbit, params.epoch_mjd_utc, next_state.t
        )
        bandit_eci = vehicle_eci_position(
            bandit_v0, params.reference_orbit, params.epoch_mjd_utc, next_state.t
        )

        # Sun ECI at current epoch + t_offset.
        epoch = _epoch_from_mjd(params.epoch_mjd_utc + float(next_state.t) / 86400.0)
        sun_eci = astrojax.sun_position(epoch)

        # Collinearity: angle between (sun→guard) and (sun→bandit).
        u_sg = guard_eci - sun_eci
        u_sb = bandit_eci - sun_eci
        u_sg = u_sg / (jnp.linalg.norm(u_sg) + 1e-12)
        u_sb = u_sb / (jnp.linalg.norm(u_sb) + 1e-12)
        cos_theta = jnp.clip(jnp.dot(u_sg, u_sb), -1.0, 1.0)
        angle_deg = jnp.rad2deg(jnp.arccos(cos_theta))

        # Ordering: bandit must be farther from sun than guard for the geometry
        # to actually block the guard's view.
        sun_to_guard_dist = jnp.linalg.norm(guard_eci - sun_eci)
        sun_to_bandit_dist = jnp.linalg.norm(bandit_eci - sun_eci)
        in_front = sun_to_bandit_dist > sun_to_guard_dist

        bandit_r = jnp.where(
            in_front,
            jnp.exp(-(angle_deg**2) / (2.0 * params.game.angle_sigma_deg**2)),
            0.0,
        )
        return jnp.where(side == Side.BANDIT, bandit_r, -bandit_r)


def make_sun_blocking(
    *,
    # Game-specific knob
    angle_sigma_deg: float = 5.0,
    # Fleet sizing
    n_guards: int = 1,
    n_bandits: int = 1,
    # Time + RNG
    dt: float = 10.0,
    max_horizon_s: float = 5400.0,  # ~90 min, full LEO orbit
    seed: int = 0,
    # Time anchor + reference frame
    epoch_mjd_utc: float = 60067.0,
    reference_orbit: ReferenceOrbitState | None = None,
    # Per-side state composition (must match dynamics frame)
    guard_components: tuple[StateComponentKey, ...] = (StateComponentKey.RTN,),
    bandit_components: tuple[StateComponentKey, ...] = (StateComponentKey.RTN,),
    # Per-side vehicle params
    guard_params: VehicleParamsSpec | None = None,
    bandit_params: VehicleParamsSpec | None = None,
    # IC sampling
    ic_sampler: ICSpec | None = None,
    # Dynamics + actuators
    truth_dynamics: DynamicsKey = DynamicsKey.HCW_RTN,
    planning_dynamics: DynamicsKey = DynamicsKey.HCW_RTN,
    guard_actuator: ActuatorKey = ActuatorKey.IMPULSIVE,
    bandit_actuator: ActuatorKey = ActuatorKey.IMPULSIVE,
    # Observation fns (None → ScenarioConfig.__post_init__ supplies FullObservation)
    guard_observation_fn: Any = None,
    bandit_observation_fn: Any = None,
):
    """Builder for an SB scenario."""
    import jax.numpy as jnp

    from orbital_game.config import ScenarioConfig, VehicleParamsSpec
    from orbital_game.reference_orbit import ReferenceOrbitState
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
        bandit_params = VehicleParamsSpec(dry_mass_kg=100.0, isp_s=220.0, max_thrust_n=5.0)
    if ic_sampler is None:
        ic_sampler = ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=500.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=2000.0,
                cross_track_m=200.0,
                along_track_offset_m=500.0,
                phase_rad=jnp.pi / 4.0,
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
        planning_dynamics=planning_dynamics,
        guard_actuator=guard_actuator,
        bandit_actuator=bandit_actuator,
        guard_observation_fn=guard_observation_fn,
        bandit_observation_fn=bandit_observation_fn,
        game=SunBlocking(angle_sigma_deg=angle_sigma_deg),
        reward_fn=SunBlockingReward(),
    )
    # Termination: max_steps only — set breach_distance to 0 so it never triggers.
    object.__setattr__(
        cfg,
        "termination_fn",
        MaxStepsOrBreach(max_steps=cfg.max_steps, breach_distance_m=0.0),
    )
    return cfg
