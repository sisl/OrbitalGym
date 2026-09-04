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

from orbitalgym.games.base import Game
from orbitalgym.registry import (
    DynamicsKey,
    GameKey,
    StateComponentKey,
    register_game,
)

if TYPE_CHECKING:
    from orbitalgym.config import VehicleParamsSpec
    from orbitalgym.reference_orbit import ReferenceOrbitState
    from orbitalgym.sampling.spec import ICSpec


@register_game(GameKey.LADY_BANDIT_GUARD)
@dataclass(frozen=True)
class LadyBanditGuard(Game):
    """Guard protects the lady (RTN origin) from bandits.

    Knobs:
        breach_radius_m: bandit-vs-lady distance below which the bandit is
            considered to have breached (bandit win).
        catch_radius_m: guard-vs-bandit distance below which the guard
            catches the bandit (guard win).
        breach_speed_mps: bandit-vs-lady relative speed below which a pass
            inside `breach_radius_m` counts as a breach. Infinite gates on
            radius alone.
        catch_speed_mps: guard-vs-bandit relative speed below which a pass
            inside `catch_radius_m` counts as a catch. Infinite gates on
            radius alone.
        escape_radius_m: bandit-vs-lady distance above which the bandit
            counts as repelled. Zero disables the gate.
        repel_on_empty_tank: when True, a bandit out of propellant whose
            coast cannot reach the lady before the horizon ends counts as
            repelled. Requires a mass-tracked bandit in an RTN frame.

    The episode is a guard win by repulsion once *every* bandit is repelled.
    """

    breach_radius_m: float = 5.0
    catch_radius_m: float = 50.0
    breach_speed_mps: float = float("inf")
    catch_speed_mps: float = float("inf")
    escape_radius_m: float = 0.0
    repel_on_empty_tank: bool = False

    def validate(self, cfg: Any) -> None:
        if not self.repel_on_empty_tank:
            return
        missing = [
            key.value
            for key in (StateComponentKey.RTN, StateComponentKey.MASS)
            if key not in cfg.bandit_components
        ]
        if missing:
            raise ValueError(
                "repel_on_empty_tank reads the bandit's propellant and coasts its "
                "RTN state to the horizon, so bandit_components must include "
                f"{', '.join(missing)}; got {tuple(k.value for k in cfg.bandit_components)}."
            )

    def default_reward_fn(self):
        from orbitalgym.rewards.lbg_zero_sum import LbgZeroSumReward

        return LbgZeroSumReward(
            catch_radius_m=self.catch_radius_m,
            breach_radius_m=self.breach_radius_m,
            catch_speed_mps=self.catch_speed_mps,
            breach_speed_mps=self.breach_speed_mps,
            escape_radius_m=self.escape_radius_m,
            repel_on_empty_tank=self.repel_on_empty_tank,
        )

    def default_termination_fn(self):
        from orbitalgym.termination.lbg_events import LbgEventTermination

        return LbgEventTermination(
            breach_radius_m=self.breach_radius_m,
            catch_radius_m=self.catch_radius_m,
            breach_speed_mps=self.breach_speed_mps,
            catch_speed_mps=self.catch_speed_mps,
            escape_radius_m=self.escape_radius_m,
            repel_on_empty_tank=self.repel_on_empty_tank,
        )


def make_lady_bandit_guard(
    *,
    # Game-specific knobs
    breach_radius_m: float = 5.0,
    catch_radius_m: float = 50.0,
    breach_speed_mps: float = float("inf"),
    catch_speed_mps: float = float("inf"),
    escape_radius_m: float = 0.0,
    repel_on_empty_tank: bool = False,
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
    # Ground-station networks (per-side; None → no comms-network gating)
    guard_ground_station_network: Any = None,
    bandit_ground_station_network: Any = None,
    # Decentralized-communication extension. When True, the guard side gains
    # the COMMUNICATE action component and the reward function charges
    # `comm_cost` per active broadcast. Bandit-side comms-leak observation
    # is wired separately by the caller (see CommsLeakObservation).
    with_communication: bool = False,
    comm_cost: float = 5.0,
    **config_kwargs: Any,
):
    """Builder for an LBG scenario.

    Wires `LadyBanditGuard(breach_radius_m=..., catch_radius_m=...)` as
    `cfg.game`; reward and termination are populated from
    `LadyBanditGuard.default_*_fn` by `ScenarioConfig.__post_init__`.

    Defaults assume a 1v1 RTN scenario with mass-tracked guard. To switch
    to 2D RT dynamics, pass matching `truth_dynamics`/`policy_dynamics`
    *and* corresponding `guard_components`/`bandit_components`; otherwise
    `ScenarioConfig.__post_init__` will reject the incoherent combo.

    Additional keyword arguments are forwarded to `ScenarioConfig`.
    """
    import jax.numpy as jnp

    from orbitalgym.config import ScenarioConfig, VehicleParamsSpec
    from orbitalgym.reference_orbit import ReferenceOrbitState
    from orbitalgym.sampling.mass import ConstantMass
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
        from orbitalgym.registry import ActionComponentKey
        from orbitalgym.rewards.lbg_with_comms import LbgWithCommsReward

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
        guard_ground_station_network=guard_ground_station_network,
        bandit_ground_station_network=bandit_ground_station_network,
        game=LadyBanditGuard(
            breach_radius_m=breach_radius_m,
            catch_radius_m=catch_radius_m,
            breach_speed_mps=breach_speed_mps,
            catch_speed_mps=catch_speed_mps,
            escape_radius_m=escape_radius_m,
            repel_on_empty_tank=repel_on_empty_tank,
        ),
        **extra_kwargs,
        **config_kwargs,
    )
    return cfg
