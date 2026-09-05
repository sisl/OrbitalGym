# Games

Typed `Game` subtypes (knob bundles) and per-game `make_<game>(...)` builders.

## Base

::: orbitalgym.games.base.Game

::: orbitalgym.games.base.NoGame

::: orbitalgym.registry.GameKey

## Lady-Bandit-Guard

::: orbitalgym.games.lady_bandit_guard.LadyBanditGuard

::: orbitalgym.games.lady_bandit_guard.make_lady_bandit_guard

The reward these knobs configure, including the `dv_cost` fuel term, is
documented under
[Extending → Customize rewards](../extending/customize-rewards.md#orbitalgym.rewards.lbg_zero_sum.LbgZeroSumReward).

## Pursuit-Evasion

::: orbitalgym.games.pursuit_evasion.PursuitEvasion

::: orbitalgym.games.pursuit_evasion.PursuitEvasionReward

::: orbitalgym.games.pursuit_evasion.PursuitEvasionTermination

::: orbitalgym.games.pursuit_evasion.make_pursuit_evasion

## Sun-Blocking

::: orbitalgym.games.sun_blocking.SunBlocking

::: orbitalgym.games.sun_blocking.SunBlockingReward

::: orbitalgym.games.sun_blocking.make_sun_blocking

## Observation-Blocking

::: orbitalgym.games.observation_blocking.ObservationBlocking

::: orbitalgym.games.observation_blocking.ObservationBlockingReward

::: orbitalgym.games.observation_blocking.make_observation_blocking

## Get-Off-My-Lawn

::: orbitalgym.games.get_off_my_lawn.GetOffMyLawn

::: orbitalgym.games.get_off_my_lawn.GetOffMyLawnReward

::: orbitalgym.games.get_off_my_lawn.GetOffMyLawnTermination

::: orbitalgym.games.get_off_my_lawn.make_get_off_my_lawn

## Proximity events

Closest-approach geometry shared by the Lady-Bandit-Guard reward and
termination, so both resolve catch and breach over the whole step.

::: orbitalgym.games.proximity.positions

::: orbitalgym.games.proximity.closest_approach

::: orbitalgym.games.proximity.proximity_event

::: orbitalgym.games.proximity.lbg_events

::: orbitalgym.games.proximity.lbg_events_from_positions

::: orbitalgym.games.proximity.stm_power_stack

::: orbitalgym.games.proximity.ballistic_breach_possible

::: orbitalgym.games.proximity.lbg_repelled

## Dispatch

::: orbitalgym.games.make_game
