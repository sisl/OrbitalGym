"""Pursuit-Evasion game.

1v1 scenario; reference orbit is the dynamical anchor only (not a
protected asset). Reward is zero-sum on relative distance:
    bandit_reward = -|r_guard - r_bandit|     (bandit wants to close)
    guard_reward  = +|r_guard - r_bandit|     (guard wants to escape)

Termination: max_steps OR capture (relative distance below threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from orbital_game.games.base import Game
from orbital_game.registry import (
    DynamicsKey,
    GameKey,
    RewardFnKey,
    StateComponentKey,
    TerminationFnKey,
    register,
    register_game,
)

if TYPE_CHECKING:
    from orbital_game.config import VehicleParamsSpec
    from orbital_game.reference_orbit import ReferenceOrbitState
    from orbital_game.sampling.spec import ICSpec


@register_game(GameKey.PURSUIT_EVASION)
@dataclass(frozen=True)
class PursuitEvasion(Game):
    """1v1 pursuit/evasion.

    Knobs:
        capture_distance_m: episode terminates when relative distance falls
            below this threshold.
    """

    capture_distance_m: float = 10.0


def _relative_position(guards, bandits) -> jnp.ndarray:
    """Returns the (1v1) guard→bandit displacement vector in RTN.

    Both sides share the same RTN frame anchored on the reference orbit;
    relative distance is just |g_rtn[:3] - b_rtn[:3]|.
    """
    if hasattr(guards, "rtn"):
        g_pos = guards.rtn[0, :3]
        b_pos = bandits.rtn[0, :3]
    else:
        g_pos = guards.rt[0, :2]
        b_pos = bandits.rt[0, :2]
    return g_pos - b_pos


@register(RewardFnKey.PURSUIT_EVASION)
@dataclass(frozen=True)
class PursuitEvasionReward:
    """Zero-sum reward on |r_guard - r_bandit|. Requires `cfg.game: PursuitEvasion`."""

    # RewardScope is a StrEnum; "per_side" is RewardScope.PER_SIDE.value.
    # Import is deferred to avoid a circular import through rewards.base ->
    # env.types -> env.__init__ -> env.core -> config -> games.__init__.
    scope: str = "per_side"

    def __call__(self, prev_state, action, next_state, side, params, t):
        del prev_state, action, t
        from orbital_game.env.types import Side

        if not isinstance(params.game, PursuitEvasion):
            raise TypeError(
                f"PursuitEvasionReward requires cfg.game: PursuitEvasion; "
                f"got {type(params.game).__name__}"
            )
        rel = _relative_position(next_state.guards, next_state.bandits)
        dist = jnp.linalg.norm(rel)
        # Bandit wants to minimize distance → reward = -dist.
        # Guard wants to maximize distance → reward = +dist.
        return jnp.where(side == Side.BANDIT, -dist, dist)


@register(TerminationFnKey.PURSUIT_EVASION)
@dataclass(frozen=True)
class PursuitEvasionTermination:
    """Episode ends on max_steps OR capture (relative distance below threshold).

    Reads max_steps from the params (cfg) and capture_distance_m from
    cfg.game.
    """

    max_steps: int

    def __call__(self, state, params, t):
        del t
        if not isinstance(params.game, PursuitEvasion):
            raise TypeError(
                f"PursuitEvasionTermination requires cfg.game: PursuitEvasion; "
                f"got {type(params.game).__name__}"
            )
        hit_max = state.step >= self.max_steps
        rel = _relative_position(state.guards, state.bandits)
        dist = jnp.linalg.norm(rel)
        captured = dist < params.game.capture_distance_m
        return jnp.logical_or(hit_max, captured)


def make_pursuit_evasion(
    *,
    # Game-specific knob
    capture_distance_m: float = 10.0,
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
    """Builder for a PE scenario."""
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
                radial_ellipse_m=1000.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=500.0,
                cross_track_m=100.0,
                along_track_offset_m=200.0,
                phase_rad=jnp.pi / 2.0,
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
        game=PursuitEvasion(capture_distance_m=capture_distance_m),
        reward_fn=PursuitEvasionReward(),
    )
    object.__setattr__(cfg, "termination_fn", PursuitEvasionTermination(max_steps=cfg.max_steps))
    return cfg
