"""Lady-Bandit-Guard game.

Guard protects the lady (a virtual point at the RTN origin) from bandit
attackers. Bandits try to reach the lady within `breach_radius_m`; the
guard tries to catch any bandit within `catch_radius_m` first. Either
condition may additionally require *dwell*: holding the radius for a number
of consecutive steps rather than brushing it once.

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
from orbitalgym.games.proximity import lbg_dwell_step
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
        catch_dwell_steps: consecutive steps a bandit must spend inside the
            catch radius of at least one guard — the same guard or not —
            before the catch fires. Zero fires on the first step that meets
            the radius and speed gates.
        breach_dwell_steps: consecutive steps a bandit must spend inside the
            breach radius of the lady before the breach fires. Zero fires on
            the first such step.
        escape_radius_m: bandit-vs-lady distance above which the bandit
            counts as repelled. Zero disables the gate.
        repel_on_empty_tank: when True, a bandit out of propellant whose
            coast cannot reach the lady before the horizon ends counts as
            repelled. Requires a mass-tracked bandit in an RTN frame.
        dv_cost: reward units charged per m/s of delta-v. Each side pays only
            for its own fuel; a side without a MASS component pays nothing.
        shaping_scale_m: distance that makes the shaping potential unity.
        shaping_gain: scale on the potential-difference shaping term. Zero
            leaves a terminal-only reward. The default is large enough that the
            shaping steers a horizon-limited planner rather than being lost
            under the terminal payoffs.
        shaping_discount: discount inside the potential difference. Matching
            the planner's per-step discount makes the shaped value exactly
            the unshaped value minus the potential.
        home_weight: weight on the guard-lady distance inside the potential,
            pulling the guard back toward the asset it defends.
        guard_separation_m: guard-guard distance below which the guard side
            pays a crowding hinge.
        lady_keepout_m: guard-lady distance below which the guard side pays
            the same hinge.
        separation_cost: reward units at the bottom of either hinge.

    The episode is a guard win by repulsion once *every* bandit is repelled.
    """

    breach_radius_m: float = 5.0
    catch_radius_m: float = 50.0
    breach_speed_mps: float = float("inf")
    catch_speed_mps: float = float("inf")
    catch_dwell_steps: int = 0
    breach_dwell_steps: int = 0
    escape_radius_m: float = 0.0
    repel_on_empty_tank: bool = False
    dv_cost: float = 0.0
    shaping_scale_m: float = 300.0
    shaping_gain: float = 50.0
    shaping_discount: float = 1.0
    home_weight: float = 0.0
    guard_separation_m: float = 20.0
    lady_keepout_m: float = 20.0
    separation_cost: float = 10.0

    def advance_state(self, prev_state: Any, next_state: Any, cfg: Any) -> Any:
        """Advance the per-bandit dwell counters onto the state leaving the step.

        Runs whatever the dwell knobs are set to, so the counters describe
        the engagement even when the win conditions do not read them.
        """
        dwell_catch, dwell_breach = lbg_dwell_step(
            prev_state,
            next_state,
            cfg.dt,
            self.catch_radius_m,
            self.catch_speed_mps,
            self.breach_radius_m,
            self.breach_speed_mps,
        )
        return next_state.replace(dwell_catch=dwell_catch, dwell_breach=dwell_breach)

    def validate(self, cfg: Any) -> None:
        for name in ("catch_dwell_steps", "breach_dwell_steps"):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(
                    f"{name} counts consecutive steps, so it must be a non-negative "
                    f"integer; got {value!r}. Zero fires the event on the first step "
                    "inside the radius."
                )
        if self.shaping_scale_m <= 0.0:
            raise ValueError(
                "shaping_scale_m divides every distance in the shaping potentials, "
                f"so it must be a positive length in metres; got {self.shaping_scale_m!r}. "
                "Set shaping_gain=0.0 to turn the shaping off instead."
            )
        if not self.repel_on_empty_tank:
            return
        if cfg.truth_dynamics is not DynamicsKey.HCW_RTN:
            raise ValueError(
                "repel_on_empty_tank coasts the bandit under the HCW-RTN state "
                "transition matrix, which only models the truth dynamics when "
                f"those are {DynamicsKey.HCW_RTN.value}; got "
                f"{getattr(cfg.truth_dynamics, 'value', cfg.truth_dynamics)}."
            )
        missing = [
            key.value
            for key in (StateComponentKey.RTN, StateComponentKey.MASS)
            if key not in cfg.bandit_components_extended
        ]
        if missing:
            raise ValueError(
                "repel_on_empty_tank reads the bandit's propellant and coasts its "
                "RTN state to the horizon, so bandit_components must include "
                f"{', '.join(missing)}; got "
                f"{tuple(k.value for k in cfg.bandit_components_extended)}."
            )

    def default_reward_fn(self):
        from orbitalgym.rewards.lbg_zero_sum import LbgZeroSumReward

        return LbgZeroSumReward(
            catch_radius_m=self.catch_radius_m,
            breach_radius_m=self.breach_radius_m,
            catch_speed_mps=self.catch_speed_mps,
            breach_speed_mps=self.breach_speed_mps,
            catch_dwell_steps=self.catch_dwell_steps,
            breach_dwell_steps=self.breach_dwell_steps,
            escape_radius_m=self.escape_radius_m,
            repel_on_empty_tank=self.repel_on_empty_tank,
            dv_cost=self.dv_cost,
            shaping_scale_m=self.shaping_scale_m,
            shaping_gain=self.shaping_gain,
            shaping_discount=self.shaping_discount,
            home_weight=self.home_weight,
            guard_separation_m=self.guard_separation_m,
            lady_keepout_m=self.lady_keepout_m,
            separation_cost=self.separation_cost,
        )

    def default_termination_fn(self):
        from orbitalgym.termination.lbg_events import LbgEventTermination

        return LbgEventTermination(
            breach_radius_m=self.breach_radius_m,
            catch_radius_m=self.catch_radius_m,
            breach_speed_mps=self.breach_speed_mps,
            catch_speed_mps=self.catch_speed_mps,
            catch_dwell_steps=self.catch_dwell_steps,
            breach_dwell_steps=self.breach_dwell_steps,
            escape_radius_m=self.escape_radius_m,
            repel_on_empty_tank=self.repel_on_empty_tank,
        )


# LBG event geometry and reward shaping the comms reward does not read, with
# the builder defaults that leave it inert. Kept beside the builder signature
# they mirror.
_COMMS_UNSUPPORTED_DEFAULTS = {
    "breach_radius_m": 5.0,
    "catch_radius_m": 50.0,
    "breach_speed_mps": float("inf"),
    "catch_speed_mps": float("inf"),
    "catch_dwell_steps": 0,
    "breach_dwell_steps": 0,
    "escape_radius_m": 0.0,
    "repel_on_empty_tank": False,
    "dv_cost": 0.0,
    "shaping_scale_m": 300.0,
    "shaping_gain": 50.0,
    "shaping_discount": 1.0,
    "home_weight": 0.0,
    "guard_separation_m": 20.0,
    "lady_keepout_m": 20.0,
    "separation_cost": 10.0,
}


def make_lady_bandit_guard(
    *,
    # Game-specific knobs
    breach_radius_m: float = 5.0,
    catch_radius_m: float = 50.0,
    breach_speed_mps: float = float("inf"),
    catch_speed_mps: float = float("inf"),
    catch_dwell_steps: int = 0,
    breach_dwell_steps: int = 0,
    escape_radius_m: float = 0.0,
    repel_on_empty_tank: bool = False,
    dv_cost: float = 0.0,
    shaping_scale_m: float = 300.0,
    shaping_gain: float = 50.0,
    shaping_discount: float = 1.0,
    home_weight: float = 0.0,
    guard_separation_m: float = 20.0,
    lady_keepout_m: float = 20.0,
    separation_cost: float = 10.0,
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

        # LbgWithCommsReward scores distance to the reference orbit plus a
        # per-broadcast charge. It reads none of the LBG event geometry and
        # carries no shaping potential, so a scenario that tunes either would
        # get a reward that ignores it while the termination still enforces
        # the event geometry.
        passed = {
            "breach_radius_m": breach_radius_m,
            "catch_radius_m": catch_radius_m,
            "breach_speed_mps": breach_speed_mps,
            "catch_speed_mps": catch_speed_mps,
            "catch_dwell_steps": catch_dwell_steps,
            "breach_dwell_steps": breach_dwell_steps,
            "escape_radius_m": escape_radius_m,
            "repel_on_empty_tank": repel_on_empty_tank,
            "dv_cost": dv_cost,
            "shaping_scale_m": shaping_scale_m,
            "shaping_gain": shaping_gain,
            "shaping_discount": shaping_discount,
            "home_weight": home_weight,
            "guard_separation_m": guard_separation_m,
            "lady_keepout_m": lady_keepout_m,
            "separation_cost": separation_cost,
        }
        for name, default in _COMMS_UNSUPPORTED_DEFAULTS.items():
            value = passed[name]
            if value != default:
                raise ValueError(
                    f"with_communication=True installs LbgWithCommsReward, which "
                    f"scores distance to the reference orbit plus a per-broadcast "
                    f"charge and carries neither LBG event geometry nor a shaping "
                    f"potential, so {name}={value!r} would not reach the reward. "
                    f"Leave {name} "
                    f"at its default {default!r}, or build the scenario with "
                    f"with_communication=False and pass an explicit reward_fn."
                )

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
            catch_dwell_steps=catch_dwell_steps,
            breach_dwell_steps=breach_dwell_steps,
            escape_radius_m=escape_radius_m,
            repel_on_empty_tank=repel_on_empty_tank,
            dv_cost=dv_cost,
            shaping_scale_m=shaping_scale_m,
            shaping_gain=shaping_gain,
            shaping_discount=shaping_discount,
            home_weight=home_weight,
            guard_separation_m=guard_separation_m,
            lady_keepout_m=lady_keepout_m,
            separation_cost=separation_cost,
        ),
        **extra_kwargs,
        **config_kwargs,
    )
    return cfg
