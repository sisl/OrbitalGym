"""Zero-sum LBG reward: potential-based shaping plus terminal events.

Both sides receive a per-step signal:

    guard:
        + shaping_gain * (shaping_discount * Phi(s') - Phi(s))
        + R_catch  * 1[catch event]                  (caught bandit terminal bonus)
        + R_catch  * 1[repelled]                     (bandits repelled terminal bonus)
        - R_breach * 1[breach event]                 (lady breached terminal penalty)
        - separation cost                            (guard-side crowding charge)

    bandit (mirror of the geometry terms):
        - shaping_gain * (shaping_discount * Phi(s') - Phi(s))
        + R_breach * 1[breach event]                 (intercepted lady terminal bonus)
        - R_catch  * 1[catch event]                  (caught by guard terminal penalty)
        - R_catch  * 1[repelled]                     (repelled terminal penalty)

The state potential, written with the guard's sign, is

    Phi(s) = (d_bandit_lady_min - d_guard_bandit_min) / shaping_scale_m
             - home_weight * d_guard_lady_min / shaping_scale_m

so Phi rises as the guard closes on a bandit, falls as a bandit closes on the
lady, and — with ``home_weight`` above zero — rewards a guard that stays near
the lady it defends. ``shaping_scale_m`` puts the potential in units of a
typical engagement distance, so the shaping reward is order one per step.

Because the shaping term is the potential difference
``g Phi(s') - Phi(s)`` and nothing else, it is potential-based shaping in the
sense of Ng, Harada and Russell (1999): the optimal policies of the shaped and
unshaped games coincide, whatever the gain, and the shaped value function is

    V'(s) = V(s) - Phi(s)

when the shaping discount matches the planner's. ``shaping_gain = 0`` removes
the term and leaves a terminal-only reward. That identity is what
:mod:`orbitalgym.policies.leaf_values` uses to keep search leaves on the
shaped scale.

The remaining two terms are deliberate per-side costs, *not* shaping, and they
do change the optimal policies:

* ``dv_cost`` per m/s of delta-v *its own* vehicles spent over the step,
  recovered from the propellant drawn down by the rocket equation. A side
  whose state carries no MASS component has no propellant trace and pays
  nothing — the term is dropped at trace time, so the reward stays
  JIT-friendly.
* the separation cost, charged to the guard side alone: a squared hinge for
  every guard pair closer than ``guard_separation_m`` and for every guard
  closer than ``lady_keepout_m`` to the lady. It buys a spread-out formation
  that does not fly through the asset it protects. A one-guard side has no
  pairs and pays only the lady term.

Neither cost is mirrored, so each side bears only what it incurs and the
geometry terms stay zero-sum.

Catch and breach events come from
:func:`orbitalgym.games.proximity.lbg_events` over the step from
``prev_state`` to ``next_state``, the same source
:class:`~orbitalgym.termination.lbg_events.LbgEventTermination` reads, so a
terminal bonus is paid in exactly the step the episode ends. The potential and
the separation cost are state functions of the step endpoints rather than of
the within-step closest approach: the shaping differences the potential at
``prev_state`` and ``next_state``, and the separation cost is charged on
``next_state``, the configuration the step arrives at.

Lady is virtual — fixed at the RTN origin (the reference orbit). The reward is
scope=PER_SIDE: returns a scalar per side and is JIT-friendly (no Python branches
on traced values).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbitalgym.actions.components import G0
from orbitalgym.env.types import Side
from orbitalgym.games.proximity import lbg_events, lbg_repelled, positions
from orbitalgym.registry import RewardFnKey, register
from orbitalgym.rewards.base import RewardScope


def lbg_potential(state, shaping_scale_m: float, home_weight: float) -> jax.Array:
    """The LBG shaping potential at ``state``, with the guard's sign.

    ``(d_bl - d_gb - home_weight * d_gl) / shaping_scale_m``, with ``d_bl`` the
    nearest bandit-lady distance, ``d_gb`` the nearest guard-bandit distance
    over all pairs, and ``d_gl`` the nearest guard-lady distance, all taken
    from the vehicle positions in ``state``. Leading axes of the side states
    pass through.
    """
    guards = positions(state.guards)
    bandits = positions(state.bandits)
    d_gb = jnp.min(
        jnp.linalg.norm(guards[..., :, None, :] - bandits[..., None, :, :], axis=-1),
        axis=(-2, -1),
    )
    d_bl = jnp.min(jnp.linalg.norm(bandits, axis=-1), axis=-1)
    d_gl = jnp.min(jnp.linalg.norm(guards, axis=-1), axis=-1)
    return (d_bl - d_gb - home_weight * d_gl) / shaping_scale_m


def _hinge(d: jax.Array, radius_m: float) -> jax.Array:
    """``(1 - d / radius)**2`` inside ``radius_m``, zero outside; zero for a nil radius."""
    if radius_m <= 0.0:
        return jnp.zeros_like(d)
    return jnp.where(d < radius_m, (1.0 - d / radius_m) ** 2, 0.0)


def guard_separation_cost(
    state,
    guard_separation_m: float,
    lady_keepout_m: float,
    separation_cost: float,
) -> jax.Array:
    """Crowding charge on the guard side at ``state``.

    One squared hinge per guard pair inside ``guard_separation_m`` plus one per
    guard inside ``lady_keepout_m`` of the lady, each scaled by
    ``separation_cost``. Returned positive; the caller subtracts it.
    """
    guards = positions(state.guards)
    pair_d = jnp.linalg.norm(guards[..., :, None, :] - guards[..., None, :, :], axis=-1)
    upper = jnp.triu(jnp.ones(pair_d.shape[-2:], dtype=bool), k=1)
    pairs = jnp.sum(jnp.where(upper, _hinge(pair_d, guard_separation_m), 0.0), axis=(-2, -1))
    lady = jnp.sum(_hinge(jnp.linalg.norm(guards, axis=-1), lady_keepout_m), axis=-1)
    return separation_cost * (pairs + lady)


@register(RewardFnKey.LBG_ZERO_SUM)
@dataclass(frozen=True)
class LbgZeroSumReward:
    """Potential-shaped zero-sum reward for the Lady-Bandit-Guard game.

    The two sides receive mirrored geometry signals: the shaping term gives a
    dense gradient toward the right behavior without moving the optimum, while
    the terminal events dominate the cumulative return when triggered.

    ``shaping_scale_m`` is the distance that makes the potential unity,
    ``shaping_gain`` scales the shaping term (zero for a terminal-only game),
    ``shaping_discount`` must match the planner's per-step discount for
    ``V' = V - Phi`` to hold exactly, and ``home_weight`` prices a guard's
    distance from the lady inside the potential.

    ``guard_separation_m``, ``lady_keepout_m`` and ``separation_cost`` set the
    guard-side crowding charge; ``dv_cost`` is the price of fuel in reward
    units per m/s of delta-v. Both are real per-side costs rather than
    shaping: each side pays only its own, so raising either makes that side
    more careful without handing the other an advantage.
    """

    shaping_scale_m: float = 300.0
    shaping_gain: float = 1.0
    shaping_discount: float = 1.0
    home_weight: float = 0.0
    guard_separation_m: float = 20.0
    lady_keepout_m: float = 20.0
    separation_cost: float = 10.0
    r_catch: float = 1000.0
    r_breach: float = 1000.0
    catch_radius_m: float = 50.0
    breach_radius_m: float = 5.0
    catch_speed_mps: float = float("inf")
    breach_speed_mps: float = float("inf")
    escape_radius_m: float = 0.0
    repel_on_empty_tank: bool = False
    dv_cost: float = 0.0
    scope: RewardScope = RewardScope.PER_SIDE

    def potential(self, state) -> jax.Array:
        """:func:`lbg_potential` at this reward's scale and home weight."""
        return lbg_potential(state, self.shaping_scale_m, self.home_weight)

    def _fuel_cost(self, prev_state, next_state, side, params):
        """Cost of the delta-v ``side`` spent over the step, in reward units.

        Delta-v follows from the propellant consumed rather than from the
        commanded impulse: thrust limits and an empty tank both clip a
        command, so the commanded magnitude would overstate the spend.
        """
        prev_side = prev_state.guards if side is Side.GUARD else prev_state.bandits
        if not hasattr(prev_side, "propellant_mass"):
            return jnp.zeros(())
        next_side = next_state.guards if side is Side.GUARD else next_state.bandits
        vehicle = params.guard_params if side is Side.GUARD else params.bandit_params
        dry = jnp.asarray(vehicle.dry_mass_kg, dtype=prev_side.propellant_mass.dtype)
        dv = (
            vehicle.isp_s
            * G0
            * jnp.log((dry + prev_side.propellant_mass) / (dry + next_side.propellant_mass))
        )
        return self.dv_cost * jnp.sum(dv)

    def __call__(self, prev_state, action, next_state, side, params, t):
        del action, t

        caught, breached, _, _ = lbg_events(
            prev_state,
            next_state,
            params.dt,
            self.catch_radius_m,
            self.catch_speed_mps,
            self.breach_radius_m,
            self.breach_speed_mps,
        )
        repelled = lbg_repelled(
            next_state,
            params,
            self.escape_radius_m,
            self.repel_on_empty_tank,
            self.breach_radius_m,
            self.breach_speed_mps,
        )
        # A repelled step pays what a catch pays: the bandit team is out of
        # the fight either way, so the guard should be indifferent between
        # intercepting a bandit and driving it off.
        guard_win = jnp.logical_or(caught, repelled).astype(jnp.float32)
        breach_event = breached.astype(jnp.float32)

        shaping = self.shaping_gain * (
            self.shaping_discount * self.potential(next_state) - self.potential(prev_state)
        )
        fuel = self._fuel_cost(prev_state, next_state, side, params)

        if side is Side.GUARD:
            separation = guard_separation_cost(
                next_state,
                self.guard_separation_m,
                self.lady_keepout_m,
                self.separation_cost,
            )
            return (
                shaping
                + self.r_catch * guard_win
                - self.r_breach * breach_event
                - separation
                - fuel
            )
        return -shaping + self.r_breach * breach_event - self.r_catch * guard_win - fuel
