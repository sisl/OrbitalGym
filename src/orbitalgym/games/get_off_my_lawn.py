"""Get-Off-My-Lawn game.

The bandit's objective is to approach the lady (reference-orbit origin)
and *loiter* within a desired bandit-to-lady standoff band
``[r_min_keep_m, r_max_keep_m]`` while avoiding interception by any
guard. The guard's objective is symmetric: intercept any bandit (within
``catch_radius_m``) or push the bandit outside ``keep_out_radius_m`` (a
larger keep-out shell tuned to be beyond ``r_max_keep_m``).

**HCW-aware loiter metric.** Bounded relative orbits in the HCW frame
trace a 2:1 ellipse in the in-plane (radial, along-track) plane:
``(x_r, x_t) = (R cos θ, -2R sin θ)``, with cross-track ``x_n`` an
independent oscillator. We measure the bandit's "standoff" not as raw
Euclidean distance to the origin (which would penalize the natural
ellipse shape) but as the radial-ellipse semi-major axis ``R`` of the
*smallest* bounded orbit passing through the bandit's current position::

    R_lady(x_r, x_t, x_n) = sqrt(x_r² + (x_t/2)² + x_n²)

This way the loiter band ``R_lady ∈ [r_min_keep, r_max_keep]`` is an
elliptical annulus aligned with the natural relative-orbit shape, and a
bandit that *holds station* on a stable orbit with ``R ∈ [r_min, r_max]``
earns the loiter bonus at every point on its orbit (not just at the
radial extrema). For RT (2D) scenarios, drop the cross-track term.

Reward (dense + terminal):

    bandit per step:
        + r_loiter * 1[R_lady ∈ [r_min_keep, r_max_keep]]   (loiter bonus)
        - alpha * d_band                                    (dense band term)
        - R_catch  * 1[d_guard_bandit_min < r_catch]        (caught penalty)

    guard per step (mirror, with extra shaping):
        - r_loiter * 1[R_lady ∈ [r_min_keep, r_max_keep]]
        + alpha * d_band
        + R_catch  * 1[d_guard_bandit_min < r_catch]
        - alpha_chase * d_guard_bandit_min                  (chase shaping)
        + R_pushout * 1[R_lady > r_keep_out]                (pushed out bonus)

Where:
    R_lady             = HCW-natural standoff (formula above)
    d_band             = max(r_min_keep - R_lady, 0) + max(R_lady - r_max_keep, 0)
                         (linear distance from the keep band; zero inside)
    d_guard_bandit_min = min over (g, b) of ||guard - bandit||  (Euclidean)

Catch is a *physical* proximity event so it stays Euclidean. Loiter and
keep-out are *orbit-shape* events and use ``R_lady``.

The loiter bonus + chase shaping are zero-sum-violating by construction
because they reward two *different* events (bandit loitering vs. guard
chasing distance). For a strictly zero-sum variant, set ``alpha_chase=0``
and ``r_pushout=0``; the catch and loiter terms remain mirrored.

Termination: max steps OR all bandits pushed beyond ``keep_out_radius_m``
(in the ``R_lady`` metric) OR any bandit caught (within ``catch_radius_m``
of any guard, Euclidean).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from orbitalgym.games.base import Game
from orbitalgym.registry import (
    DynamicsKey,
    GameKey,
    RewardFnKey,
    StateComponentKey,
    TerminationFnKey,
    register,
    register_game,
)

if TYPE_CHECKING:
    from orbitalgym.config import VehicleParamsSpec
    from orbitalgym.reference_orbit import ReferenceOrbitState
    from orbitalgym.sampling.spec import ICSpec


def _positions(side_state):
    """Per-vehicle position vectors from a side state pytree (RTN→RT fallback)."""
    if hasattr(side_state, "rtn"):
        return side_state.rtn[:, :3]
    return side_state.rt[:, :2]


def _hcw_lady_distance(positions: jnp.ndarray) -> jnp.ndarray:
    """HCW-natural standoff: the radial-ellipse semi-major axis ``R`` of the
    smallest bounded relative orbit passing through ``positions``.

    For an HCW bounded relative orbit centered on the lady (no along-track
    offset, no drift), in-plane motion is an ellipse with radial amplitude
    ``R`` and along-track amplitude ``2R``; cross-track is an independent
    1:1 oscillator with amplitude ``N``. Given a position ``(x_r, x_t, x_n)``,
    the *smallest* in-plane orbit reaching ``(x_r, x_t)`` has
    ``R = sqrt(x_r² + (x_t/2)²)``. Treating cross-track as 1:1, we compose::

        R_lady = sqrt(x_r² + (x_t/2)² + x_n²)

    This makes the loiter band an in-plane elliptical annulus that aligns
    with the natural relative-orbit shape — a bandit holding station on a
    stable bounded orbit with semi-major axis in ``[r_min, r_max]`` earns
    the loiter bonus at every point on its orbit, not just at the radial
    apsides.

    Accepts shape ``(..., 2)`` (RT) or ``(..., 3)`` (RTN). Returns shape
    ``(...,)``.
    """
    x_r = positions[..., 0]
    x_t = positions[..., 1]
    in_plane_sq = x_r * x_r + (x_t * 0.5) * (x_t * 0.5)
    if positions.shape[-1] >= 3:
        x_n = positions[..., 2]
        return jnp.sqrt(in_plane_sq + x_n * x_n)
    return jnp.sqrt(in_plane_sq)


@register_game(GameKey.GET_OFF_MY_LAWN)
@dataclass(frozen=True)
class GetOffMyLawn(Game):
    """Bandit wants to loiter near the lady; guard wants to intercept or repel.

    Geometric knobs:
        r_min_keep_m: lower bound of bandit desired loiter band (m, HCW metric).
        r_max_keep_m: upper bound of bandit desired loiter band (m, HCW metric).
        catch_radius_m: guard-vs-bandit Euclidean distance below which the
            guard intercepts (terminal event).
        keep_out_radius_m: HCW standoff beyond which the bandit is considered
            "pushed out". Must be ``> r_max_keep_m`` so the keep-out shell
            sits outside the loiter band; ``__post_init__`` enforces this.

    Reward-weight knobs (tune which term dominates):
        r_loiter: per-step bonus when at least one bandit is inside the keep
            band. Default 1.0.
        alpha: dense band-distance shaping coefficient (per meter). Default
            ``1e-3`` — a 1000 m gap from the band costs ~1.0 per step.
        r_catch: terminal penalty on the bandit (and bonus on the guard) when
            the guard catches a bandit. Default 1000.0; raise to make catch
            dominate, lower to weight loiter / chase shaping more heavily.
        r_pushout: bonus on the guard when every bandit is beyond the
            keep-out shell. Default 500.0.
        alpha_chase: dense guard-side shaping per meter of guard-bandit
            distance (a chase prior). Default ``1e-3``. Set to 0.0 (along
            with ``r_pushout=0.0``) for a strictly zero-sum reward.
    """

    r_min_keep_m: float = 100.0
    r_max_keep_m: float = 500.0
    catch_radius_m: float = 50.0
    keep_out_radius_m: float = 1500.0
    r_loiter: float = 1.0
    alpha: float = 1e-3
    r_catch: float = 1000.0
    r_pushout: float = 500.0
    alpha_chase: float = 1e-3

    def __post_init__(self) -> None:
        if not (self.r_min_keep_m < self.r_max_keep_m):
            raise ValueError(
                f"GetOffMyLawn: r_min_keep_m ({self.r_min_keep_m}) must be < "
                f"r_max_keep_m ({self.r_max_keep_m})."
            )
        if not (self.keep_out_radius_m > self.r_max_keep_m):
            raise ValueError(
                f"GetOffMyLawn: keep_out_radius_m ({self.keep_out_radius_m}) must be "
                f"strictly larger than r_max_keep_m ({self.r_max_keep_m}); the keep-out "
                f"shell must sit outside the loiter band."
            )
        if self.catch_radius_m < 0.0:
            raise ValueError(
                f"GetOffMyLawn: catch_radius_m must be >= 0; got {self.catch_radius_m}"
            )

    def default_reward_fn(self):
        return GetOffMyLawnReward(
            r_min_keep_m=self.r_min_keep_m,
            r_max_keep_m=self.r_max_keep_m,
            catch_radius_m=self.catch_radius_m,
            keep_out_radius_m=self.keep_out_radius_m,
            alpha=self.alpha,
            alpha_chase=self.alpha_chase,
            r_loiter=self.r_loiter,
            r_catch=self.r_catch,
            r_pushout=self.r_pushout,
        )

    def default_termination_fn(self):
        return GetOffMyLawnTermination(
            catch_radius_m=self.catch_radius_m,
            keep_out_radius_m=self.keep_out_radius_m,
        )


@register(RewardFnKey.GET_OFF_MY_LAWN)
@dataclass(frozen=True)
class GetOffMyLawnReward:
    """Dense + terminal reward for Get-Off-My-Lawn.

    See :class:`GetOffMyLawn` module docstring for the full formula.
    """

    r_min_keep_m: float = 100.0
    r_max_keep_m: float = 500.0
    catch_radius_m: float = 50.0
    keep_out_radius_m: float = 1500.0
    alpha: float = 1e-3
    alpha_chase: float = 1e-3
    r_loiter: float = 1.0
    r_catch: float = 1000.0
    r_pushout: float = 500.0
    scope: str = "per_side"

    def __call__(self, prev_state, action, next_state, side, params, t):
        del prev_state, action, t
        from orbitalgym.env.types import Side

        if not isinstance(params.game, GetOffMyLawn):
            raise TypeError(
                f"GetOffMyLawnReward requires cfg.game: GetOffMyLawn; "
                f"got {type(params.game).__name__}"
            )

        guard_pos = _positions(next_state.guards)  # (n_g, dim)
        bandit_pos = _positions(next_state.bandits)  # (n_b, dim)

        # Pairwise guard-bandit distances (Euclidean — physical proximity).
        diffs = guard_pos[:, None, :] - bandit_pos[None, :, :]
        d_gb = jnp.linalg.norm(diffs, axis=-1)
        d_gb_min = jnp.min(d_gb)

        # HCW-natural bandit-to-lady standoff: (n_b,).
        r_lady = _hcw_lady_distance(bandit_pos)

        # "Distance to the keep band" in the HCW metric — zero inside, linear outside.
        below = jnp.maximum(self.r_min_keep_m - r_lady, 0.0)
        above = jnp.maximum(r_lady - self.r_max_keep_m, 0.0)
        d_band = below + above  # per bandit
        d_band_sum = jnp.sum(d_band)

        # Loiter event: any bandit currently inside the keep band.
        in_band = jnp.logical_and(r_lady >= self.r_min_keep_m, r_lady <= self.r_max_keep_m).astype(
            jnp.float32
        )
        loiter_event = jnp.max(in_band)  # 1.0 if at least one bandit loitering

        catch_event = (d_gb_min < self.catch_radius_m).astype(jnp.float32)

        # Pushout event: ALL bandits beyond the keep-out shell (HCW metric).
        pushed_out_per_bandit = (r_lady > self.keep_out_radius_m).astype(jnp.float32)
        pushout_event = jnp.min(pushed_out_per_bandit)  # 1.0 only if every bandit is out

        bandit_r = (
            self.r_loiter * loiter_event - self.alpha * d_band_sum - self.r_catch * catch_event
        )
        guard_r = (
            -self.r_loiter * loiter_event
            + self.alpha * d_band_sum
            + self.r_catch * catch_event
            - self.alpha_chase * d_gb_min
            + self.r_pushout * pushout_event
        )
        return jnp.where(side == Side.BANDIT, bandit_r, guard_r)


@register(TerminationFnKey.GET_OFF_MY_LAWN)
@dataclass(frozen=True)
class GetOffMyLawnTermination:
    """Terminate on max_steps OR any bandit caught OR all bandits pushed out.

    `catch_radius_m=0` disables the catch gate. `keep_out_radius_m` should
    match :class:`GetOffMyLawn` to keep reward and termination in agreement.
    """

    catch_radius_m: float
    keep_out_radius_m: float

    def __call__(self, state, params, t) -> jax.Array:
        del t
        if not isinstance(params.game, GetOffMyLawn):
            raise TypeError(
                f"GetOffMyLawnTermination requires cfg.game: GetOffMyLawn; "
                f"got {type(params.game).__name__}"
            )
        hit_max = state.step >= params.max_steps

        bandit_pos = _positions(state.bandits)
        r_lady = _hcw_lady_distance(bandit_pos)
        all_pushed_out = jnp.all(r_lady > self.keep_out_radius_m)

        if self.catch_radius_m > 0.0:
            guard_pos = _positions(state.guards)
            diffs = guard_pos[:, None, :] - bandit_pos[None, :, :]
            d_gb_min = jnp.min(jnp.linalg.norm(diffs, axis=-1))
            caught = d_gb_min < self.catch_radius_m
        else:
            caught = jnp.asarray(False)

        return jnp.logical_or(hit_max, jnp.logical_or(caught, all_pushed_out))


def make_get_off_my_lawn(
    *,
    # Geometric knobs
    r_min_keep_m: float = 100.0,
    r_max_keep_m: float = 500.0,
    catch_radius_m: float = 50.0,
    keep_out_radius_m: float = 1500.0,
    # Reward-weight knobs (tune which term dominates)
    r_loiter: float = 1.0,
    alpha: float = 1e-3,
    r_catch: float = 1000.0,
    r_pushout: float = 500.0,
    alpha_chase: float = 1e-3,
    # Fleet sizing
    n_guards: int = 1,
    n_bandits: int = 1,
    # Time + RNG
    dt: float = 10.0,
    max_horizon_s: float = 3000.0,
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
    """Builder for a Get-Off-My-Lawn scenario.

    Defaults place a 1v1 bandit/guard in the RTN frame around a circular
    LEO reference orbit. The bandit IC sits on a 2 km radial-ellipse
    co-orbit (well outside the keep band), so a passive bandit will not
    naturally drift into the loiter window — solving requires control.
    """
    import jax.numpy as jnp

    from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
    from orbitalgym.reference_orbit import ReferenceOrbitState
    from orbitalgym.sampling.side import RelativeEllipse
    from orbitalgym.sampling.spec import ICSpec

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
        # Default IC keeps the bandit comfortably inside the keep-out shell but
        # outside the loiter band — solving the game requires control on both
        # sides; a passive bandit will not naturally settle into the loiter
        # band, and a passive guard will not naturally intercept.
        ic_sampler = ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=300.0,
                cross_track_m=0.0,
                along_track_offset_m=0.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=20.0,
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=600.0,
                cross_track_m=50.0,
                along_track_offset_m=200.0,
                phase_rad=jnp.pi,
                sigma_radial_ellipse_m=20.0,
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
        game=GetOffMyLawn(
            r_min_keep_m=r_min_keep_m,
            r_max_keep_m=r_max_keep_m,
            catch_radius_m=catch_radius_m,
            keep_out_radius_m=keep_out_radius_m,
            r_loiter=r_loiter,
            alpha=alpha,
            r_catch=r_catch,
            r_pushout=r_pushout,
            alpha_chase=alpha_chase,
        ),
    )
    return cfg
