"""Observation-Blocking game.

Bandit blocks guard's view of an Earth surface target — but only when the
target is observable (guard above min_elevation_deg from the target).

Reward (PER_SIDE, zero-sum) — KSP-DG-aligned shape, gated on visibility:

    bandit_r = (-û_BG · û_BT) * exp(-decay * (d_BG - d_target)²) * 1[visible]
    guard_r  = -bandit_r

where û_BT is the unit vector from bandit to the surface target (rotated
ECEF→ECI per timestep). Visibility is computed from geocentric "up" at the
target — see the spec's "Risks" section for accuracy notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import astrojax
import jax
import jax.numpy as jnp
import numpy as np

from orbital_game.games._frames import vehicle_eci_position
from orbital_game.games.base import Game
from orbital_game.registry import (
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

    Delegates to the same helper in sun_blocking — see that module for the
    rationale (Epoch.__init__ only accepts (year, month, day, ...) or string).
    """
    from orbital_game.games.sun_blocking import _epoch_from_mjd as _sb_epoch_from_mjd

    return _sb_epoch_from_mjd(mjd_float)


def _target_eci_at_mjd(mjd: jax.Array, target_ecef: jax.Array) -> jax.Array:
    """Target ECI position at a given MJD. JIT/scan-safe via `jax.pure_callback`.

    `astrojax.rotation_ecef_to_eci` builds an `Epoch` via `_from_internal`,
    which uses `math.floor` on a Python float — incompatible with traced JAX
    scalars. We push the rotation construction (and the ECEF→ECI rotation
    itself) to the host via `pure_callback` so the reward fn is safe to use
    inside `jax.lax.scan` / `jax.jit`.
    """

    def host_fn(mjd_arr, target_ecef_arr):
        epoch = _epoch_from_mjd(float(mjd_arr))
        eop = astrojax.zero_eop()
        rot = np.asarray(astrojax.rotation_ecef_to_eci(eop, epoch), dtype=np.float64)
        return rot @ np.asarray(target_ecef_arr, dtype=np.float64)

    return jax.pure_callback(
        host_fn,
        jax.ShapeDtypeStruct((3,), jnp.float64),
        mjd,
        target_ecef,
    )


def target_visible_from_guard(
    target_eci: jax.Array,
    guard_eci: jax.Array,
    min_elevation_deg: float,
) -> jax.Array:
    """Boolean visibility gate — True if elevation(target → guard) ≥ min_elevation_deg.

    Uses geocentric "up" at the target (radial from Earth center). Dot product
    is rotation-invariant, so the choice of frame doesn't matter; the only
    approximation is geocentric vs. geodetic up. See spec.
    """
    up = target_eci / (jnp.linalg.norm(target_eci) + 1e-12)
    dir_to_guard = (guard_eci - target_eci) / (jnp.linalg.norm(guard_eci - target_eci) + 1e-12)
    sin_el = jnp.dot(up, dir_to_guard)
    elevation_rad = jnp.arcsin(jnp.clip(sin_el, -1.0, 1.0))
    elevation_deg = jnp.rad2deg(elevation_rad)
    return elevation_deg >= min_elevation_deg


def observation_blocking_kernel(
    guard_eci: jax.Array,
    bandit_eci: jax.Array,
    target_eci: jax.Array,
    target_viewing_distance_m: float,
    range_decay_coef: float,
    min_elevation_deg: float,
) -> jax.Array:
    """Pure OB reward formula. Returns the bandit's reward as a scalar.

    Same KSP-DG-style core as SB (vertex at the bandit), but multiplied by a
    visibility gate driven by the target's elevation as seen from the guard.

    Output: scalar reward in [-1, +1].
    """
    rel_bg = guard_eci - bandit_eci
    rel_bt = target_eci - bandit_eci
    d_bg = jnp.linalg.norm(rel_bg)
    u_bg = rel_bg / (d_bg + 1e-12)
    u_bt = rel_bt / (jnp.linalg.norm(rel_bt) + 1e-12)
    angular = -jnp.dot(u_bg, u_bt)
    range_factor = jnp.exp(-range_decay_coef * (d_bg - target_viewing_distance_m) ** 2)
    visible = target_visible_from_guard(target_eci, guard_eci, min_elevation_deg)
    return jnp.where(visible, angular * range_factor, 0.0)


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
        target_viewing_distance_m: desired bandit-guard standoff at which the
            range factor peaks (meters).
        range_decay_coef: Gaussian decay coefficient (1/m²) controlling the
            range-factor peak width.

    target_ecef_m is precomputed in __post_init__ (one-time geodetic→ECEF
    conversion) and cached on the instance for use in the reward fn.
    """

    target_lat_deg: float = 37.4
    target_lon_deg: float = -122.2
    target_alt_m: float = 0.0
    min_elevation_deg: float = 5.0
    target_viewing_distance_m: float = 500.0
    range_decay_coef: float = 4.0e-6
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

    def default_reward_fn(self):
        return ObservationBlockingReward()

    def default_termination_fn(self):
        from orbital_game.termination.reference import MaxStepsOnly

        return MaxStepsOnly()


@register(RewardFnKey.OBSERVATION_BLOCKING)
@dataclass(frozen=True)
class ObservationBlockingReward:
    """KSP-DG-style observation-blocking reward, gated on visibility.

    Requires `cfg.game: ObservationBlocking`. Pulls vehicle ECI positions via
    `games/_frames.vehicle_eci_position`, rotates the target ECEF→ECI per
    timestep (GMST-only via `astrojax.zero_eop`), then calls
    `observation_blocking_kernel`.
    """

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

        mjd = params.epoch_mjd_utc + next_state.t / 86400.0
        target_eci = _target_eci_at_mjd(mjd, params.game.target_ecef_m)

        bandit_r = observation_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=bandit_eci,
            target_eci=target_eci,
            target_viewing_distance_m=params.game.target_viewing_distance_m,
            range_decay_coef=params.game.range_decay_coef,
            min_elevation_deg=params.game.min_elevation_deg,
        )
        return jnp.where(side == Side.BANDIT, bandit_r, -bandit_r)


def make_observation_blocking(
    *,
    # Game-specific knobs
    target_lat_deg: float = 37.4,
    target_lon_deg: float = -122.2,
    target_alt_m: float = 0.0,
    min_elevation_deg: float = 5.0,
    target_viewing_distance_m: float = 500.0,
    range_decay_coef: float = 4.0e-6,
    # Fleet sizing
    n_guards: int = 1,
    n_bandits: int = 1,
    # Time + RNG
    dt: float = 10.0,
    max_horizon_s: float = 5400.0,
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
    # Dynamics
    truth_dynamics: DynamicsKey = DynamicsKey.HCW_RTN,
    policy_dynamics: DynamicsKey = DynamicsKey.HCW_RTN,
    # Observation fns (None → ScenarioConfig.__post_init__ supplies FullObservation)
    guard_observation_fn: Any = None,
    bandit_observation_fn: Any = None,
):
    """Builder for an OB scenario."""
    import jax.numpy as jnp

    from orbital_game.config import ScenarioConfig, VehicleParamsSpec
    from orbital_game.reference_orbit import ReferenceOrbitState
    from orbital_game.sampling.side import RelativeEllipse
    from orbital_game.sampling.spec import ICSpec

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
                radial_ellipse_m=1500.0,
                cross_track_m=100.0,
                along_track_offset_m=300.0,
                phase_rad=jnp.pi / 6.0,
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
        guard_observation_fn=guard_observation_fn,
        bandit_observation_fn=bandit_observation_fn,
        game=ObservationBlocking(
            target_lat_deg=target_lat_deg,
            target_lon_deg=target_lon_deg,
            target_alt_m=target_alt_m,
            min_elevation_deg=min_elevation_deg,
            target_viewing_distance_m=target_viewing_distance_m,
            range_decay_coef=range_decay_coef,
        ),
    )
    return cfg
