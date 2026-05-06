"""Lady-Bandit-Guard game.

Guard protects the lady (a virtual point at the RTN origin) from bandit
attackers. Bandits try to reach the lady within `breach_radius_m`; the
guard tries to catch any bandit within `catch_radius_m` first.

`LadyBanditGuard` is a typed knob bundle stored as `cfg.game`. It owns
the default reward (`LbgZeroSumReward`) and termination
(`LbgEventTermination`), which `ScenarioConfig.__post_init__` wires onto
the cfg automatically. Custom reward/termination can still be injected
via the `reward_fn=` / `termination_fn=` constructor kwargs on
`ScenarioConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from orbital_game.games.base import Game
from orbital_game.registry import (
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
    """Guard protects the lady (RTN origin) from bandits.

    Knobs:
        breach_radius_m: bandit-vs-lady distance below which the bandit is
            considered to have breached (bandit win).
        catch_radius_m: guard-vs-bandit distance below which the guard
            catches the bandit (guard win).
    """

    breach_radius_m: float = 5.0
    catch_radius_m: float = 50.0

    def default_reward_fn(self):
        from orbital_game.rewards.lbg_zero_sum import LbgZeroSumReward

        return LbgZeroSumReward(
            catch_radius_m=self.catch_radius_m,
            breach_radius_m=self.breach_radius_m,
        )

    def default_termination_fn(self):
        from orbital_game.termination.lbg_events import LbgEventTermination

        return LbgEventTermination(
            breach_radius_m=self.breach_radius_m,
            catch_radius_m=self.catch_radius_m,
        )


def make_lady_bandit_guard(
    *,
    # Game-specific knobs
    breach_radius_m: float = 5.0,
    catch_radius_m: float = 50.0,
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
    # Dynamics
    truth_dynamics: DynamicsKey = DynamicsKey.HCW_RTN,
    policy_dynamics: DynamicsKey = DynamicsKey.HCW_RTN,
    # Observation fns (None → ScenarioConfig.__post_init__ supplies FullObservation)
    guard_observation_fn: Any = None,
    bandit_observation_fn: Any = None,
    # Decentralized-communication extension. When True, the guard side gains
    # the COMMUNICATE action component and the reward function charges
    # `comm_cost` per active broadcast. Bandit-side comms-leak observation
    # is wired separately by the caller (see CommsLeakObservation).
    with_communication: bool = False,
    comm_cost: float = 5.0,
):
    """Builder for an LBG scenario.

    Wires `LadyBanditGuard(breach_radius_m=..., catch_radius_m=...)` as
    `cfg.game`; reward and termination are populated from
    `LadyBanditGuard.default_*_fn` by `ScenarioConfig.__post_init__`.

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

    extra_kwargs: dict[str, Any] = {}
    if with_communication:
        from orbital_game.registry import ActionComponentKey
        from orbital_game.rewards.lbg_with_comms import LbgWithCommsReward

        extra_kwargs["guard_action_components"] = (
            ActionComponentKey.IMPULSIVE_MANEUVER,
            ActionComponentKey.COMMUNICATE,
        )
        extra_kwargs["reward_fn"] = LbgWithCommsReward(comm_cost=comm_cost)

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
        game=LadyBanditGuard(
            breach_radius_m=breach_radius_m,
            catch_radius_m=catch_radius_m,
        ),
        **extra_kwargs,
    )
    return cfg
