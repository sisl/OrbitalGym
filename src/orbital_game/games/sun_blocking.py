"""Sun-Blocking game.

Bandit is rewarded for occluding the guard's view of the Sun. Reward is the
KSP-DG SB1 formulation:

    bandit_r = -û_BG · û_BS  *  exp(-decay * (d_BG - d_target)²)
    guard_r  = -bandit_r

where û_BG is the unit vector from bandit to guard, û_BS is the unit vector
from bandit to sun, d_BG is the bandit-guard distance, d_target is the
desired viewing distance, and `decay` is a Gaussian decay coefficient.

Reward is bounded `[-1, +1]`. Peak (`+1`) when the bandit is directly between
sun and guard at the desired range; trough (`-1`) when the guard is between
sun and bandit at the desired range; zero far from either configuration.
Zero-sum across sides.

Reference: https://github.com/mit-ll/spacegym-kspdg/blob/main/src/kspdg/sb1/sb1_base.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import astrojax
import jax
import jax.numpy as jnp
import numpy as np

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


def _sun_eci_at_mjd(mjd: jax.Array) -> jax.Array:
    """Sun ECI position at a given MJD. JIT/scan-safe via `jax.pure_callback`.

    `astrojax.sun_position` builds an `Epoch` via `_from_internal`, which uses
    `math.floor` on a Python float — incompatible with traced JAX scalars.
    We push that work to the host via `pure_callback`, which receives a
    concrete numpy scalar at execution time even when the surrounding code is
    being traced (e.g. inside `jax.lax.scan`).
    """

    def host_fn(mjd_arr):
        epoch = _epoch_from_mjd(float(mjd_arr))
        return np.asarray(astrojax.sun_position(epoch), dtype=np.float64)

    return jax.pure_callback(
        host_fn,
        jax.ShapeDtypeStruct((3,), jnp.float64),
        mjd,
    )


def sun_blocking_kernel(
    guard_eci: jax.Array,
    bandit_eci: jax.Array,
    sun_eci: jax.Array,
    target_viewing_distance_m: float,
    range_decay_coef: float,
) -> jax.Array:
    """Pure SB reward formula. Returns the bandit's reward as a scalar.

    Inputs:
        guard_eci, bandit_eci, sun_eci: (3,) position vectors in ECI [m].
        target_viewing_distance_m: peak of the range factor.
        range_decay_coef: Gaussian decay coefficient (1/m²).

    Output: scalar reward in [-1, +1].
    """
    rel_bg = guard_eci - bandit_eci
    rel_bs = sun_eci - bandit_eci
    d_bg = jnp.linalg.norm(rel_bg)
    u_bg = rel_bg / (d_bg + 1e-12)
    u_bs = rel_bs / (jnp.linalg.norm(rel_bs) + 1e-12)
    angular = -jnp.dot(u_bg, u_bs)
    range_factor = jnp.exp(-range_decay_coef * (d_bg - target_viewing_distance_m) ** 2)
    return angular * range_factor


@register_game(GameKey.SUN_BLOCKING)
@dataclass(frozen=True)
class SunBlocking(Game):
    """1v1 sun-blocking game.

    Knobs:
        target_viewing_distance_m: desired bandit-guard standoff at which the
            range factor peaks (meters).
        range_decay_coef: Gaussian decay coefficient (1/m²) controlling the
            range-factor peak width.
    """

    target_viewing_distance_m: float = 500.0
    range_decay_coef: float = 4.0e-6


@register(RewardFnKey.SUN_BLOCKING)
@dataclass(frozen=True)
class SunBlockingReward:
    """KSP-DG-style sun-blocking reward.

    Requires `cfg.game: SunBlocking`. Pulls vehicle ECI positions via
    `games/_frames.vehicle_eci_position`, looks up Sun ECI via
    `astrojax.sun_position`, then calls `sun_blocking_kernel`.
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

        guard_v0 = jax.tree_util.tree_map(lambda x: x[0], next_state.guards)
        bandit_v0 = jax.tree_util.tree_map(lambda x: x[0], next_state.bandits)

        guard_eci = vehicle_eci_position(
            guard_v0, params.reference_orbit, params.epoch_mjd_utc, next_state.t
        )
        bandit_eci = vehicle_eci_position(
            bandit_v0, params.reference_orbit, params.epoch_mjd_utc, next_state.t
        )

        mjd = params.epoch_mjd_utc + next_state.t / 86400.0
        sun_eci = _sun_eci_at_mjd(mjd)

        bandit_r = sun_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=bandit_eci,
            sun_eci=sun_eci,
            target_viewing_distance_m=params.game.target_viewing_distance_m,
            range_decay_coef=params.game.range_decay_coef,
        )
        return jnp.where(side == Side.BANDIT, bandit_r, -bandit_r)


def make_sun_blocking(
    *,
    # Game-specific knobs
    target_viewing_distance_m: float = 500.0,
    range_decay_coef: float = 4.0e-6,
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
    policy_dynamics: DynamicsKey = DynamicsKey.HCW_RTN,
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
        policy_dynamics=policy_dynamics,
        guard_actuator=guard_actuator,
        bandit_actuator=bandit_actuator,
        guard_observation_fn=guard_observation_fn,
        bandit_observation_fn=bandit_observation_fn,
        game=SunBlocking(
            target_viewing_distance_m=target_viewing_distance_m,
            range_decay_coef=range_decay_coef,
        ),
        reward_fn=SunBlockingReward(),
    )
    object.__setattr__(
        cfg,
        "termination_fn",
        MaxStepsOrBreach(max_steps=cfg.max_steps, breach_distance_m=0.0),
    )
    return cfg
