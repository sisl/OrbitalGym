"""LBG reward: per-side potential shaping plus zero-sum terminal events.

Both sides receive a per-step signal:

    guard:
        + shaping_gain * (shaping_discount * Phi_g(s') - Phi_g(s))
        + R_catch  * 1[catch event]                  (caught bandit terminal bonus)
        + R_catch  * 1[repelled]                     (bandits repelled terminal bonus)
        - R_breach * 1[breach event]                 (lady breached terminal penalty)
        - separation cost                            (guard-side crowding charge)

    bandit:
        + shaping_gain * (shaping_discount * Phi_b(s') - Phi_b(s))
        + R_breach * 1[breach event]                 (intercepted lady terminal bonus)
        - R_catch  * 1[catch event]                  (caught by guard terminal penalty)
        - R_catch  * 1[repelled]                     (repelled terminal penalty)

Each side has its own potential, expressing what that side is trying to close:

    Phi_g(s) = (-d_guard_bandit_min - home_weight * d_guard_lady_min)
               / shaping_scale_m
    Phi_b(s) = -d_bandit_lady_min / shaping_scale_m
    Phi_side(s) = 0                                   (s absorbing)

so the guard's potential rises as it closes on the nearest bandit and — with
``home_weight`` above zero — as it stays near the lady it defends, while the
bandit's rises as it closes on the lady. ``shaping_scale_m`` puts both in units
of a typical engagement distance, so the shaping reward is order one per step.

A single zero-sum potential ``(d_bl - d_gb) / L`` looks tidier but is flat
exactly where it matters: with the guard parked near the lady, a bandit run at
the lady shortens ``d_bl`` and ``d_gb`` by nearly the same amount, so the
difference barely moves and the bandit gets no gradient at all. Per-side
potentials do not have that cancellation.

Because each side's shaping term is that side's potential difference
``g Phi(s') - Phi(s)`` and nothing else, this is potential-based shaping in the
sense of Ng, Harada and Russell (1999), applied per agent. Devlin and Kudenko
("Theoretical considerations of potential-based reward shaping for multi-agent
systems", AAMAS 2011) show that giving each agent its own potential in a
multi-agent setting leaves the Nash equilibria of the underlying game
unchanged, whatever the potentials and gains, and that a side's shaped value is
its unshaped value minus its own potential. ``shaping_gain = 0`` removes the
term and leaves a terminal-only game.

The potential is zero at an absorbing state, which is the remaining condition
those results ask of an episodic game: on the step that terminates — a catch, a
breach, the bandits repelled, or the game time limit — the shaping pays ``-gain * Phi(s)`` and
nothing more, so a whole episode's shaping sums to ``-gain * Phi(s_0)``, a
constant of the initial state, with no residual ``gain * g^T * Phi(s_T)`` left
to bias which terminal state a side steers toward.

The shaping is *not* zero-sum between the sides: the two potentials are
unrelated functions and their differences do not cancel, so both sides can gain
on the same step. The terminal payoffs remain exactly zero-sum, which is what
makes the game a zero-sum game; the shaping only redistributes each side's own
return along the path to the same equilibria.

A search does not want the value identity applied at its leaves: subtracting a
side's potential there would cancel the shaping that telescoped along the path
and leave the ranking with the terminal estimate alone.
:mod:`orbitalgym.policies.leaf_values` uses the value-initialization
equivalence instead and *adds* each side's own potential, so it guides the
search without accumulating along a path.

Two further terms are deliberate per-side costs, not shaping, and they do move
the equilibria:

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

Neither cost is mirrored, so each side bears only what it incurs.

Catch and breach events come from
:func:`orbitalgym.games.proximity.lbg_events_with_dwell` over the step from
``prev_state`` to ``next_state``, the same source
:class:`~orbitalgym.termination.lbg_events.LbgEventTermination` reads —
including the dwell counters, so a bonus that waits on a dwell waits in both
places — so a terminal bonus is paid in exactly the step the episode ends, and the same
booleans are what make ``next_state`` absorbing for the shaping. The potentials
and the separation cost are otherwise state functions of the step endpoints
rather than of the within-step closest approach: the shaping differences a
potential at ``prev_state`` and ``next_state``, and the separation cost is
charged on ``next_state``, the configuration the step arrives at.

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
from orbitalgym.games.proximity import lbg_events_with_dwell, lbg_repelled, positions
from orbitalgym.registry import RewardFnKey, register
from orbitalgym.rewards.base import RewardScope


def lbg_distances(state) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(d_gb, d_bl, d_gl)`` minima from the vehicle positions in ``state``.

    ``d_gb`` is the nearest guard-bandit distance over all pairs, ``d_bl`` the
    nearest bandit-lady distance, and ``d_gl`` the nearest guard-lady distance.
    Leading axes of the side states pass through. Both potentials and any value
    estimate built on them read these three numbers, so computing them once and
    passing them to :func:`lbg_potential_from_distances` avoids repeating the
    pairwise reduction.
    """
    guards = positions(state.guards)
    bandits = positions(state.bandits)
    d_gb = jnp.min(
        jnp.linalg.norm(guards[..., :, None, :] - bandits[..., None, :, :], axis=-1),
        axis=(-2, -1),
    )
    d_bl = jnp.min(jnp.linalg.norm(bandits, axis=-1), axis=-1)
    d_gl = jnp.min(jnp.linalg.norm(guards, axis=-1), axis=-1)
    return d_gb, d_bl, d_gl


def lbg_potential_from_distances(
    side: Side,
    d_gb: jax.Array,
    d_bl: jax.Array,
    d_gl: jax.Array,
    shaping_scale_m: float,
    home_weight: float,
) -> jax.Array:
    """The potential ``side`` sees, from distances already reduced.

    The guard's is ``(-d_gb - home_weight * d_gl) / shaping_scale_m`` and the
    bandit's is ``-d_bl / shaping_scale_m``, so each side's potential rises as
    that side closes what it is chasing. ``home_weight`` reaches the guard's
    potential only; the bandit's does not read where the guard is.
    """
    if side is Side.BANDIT:
        return -d_bl / shaping_scale_m
    return (-d_gb - home_weight * d_gl) / shaping_scale_m


def lbg_potential(
    state,
    side: Side,
    shaping_scale_m: float,
    home_weight: float,
) -> jax.Array:
    """The shaping potential ``side`` sees at ``state``.

    :func:`lbg_potential_from_distances` applied to :func:`lbg_distances`. The
    potential is defined as zero at an absorbing state; the reward applies that
    on the terminating transition rather than here, because whether a step
    terminates is a property of the transition, not of the arriving state alone.
    """
    return lbg_potential_from_distances(side, *lbg_distances(state), shaping_scale_m, home_weight)


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
    """Potential-shaped reward with zero-sum terminal events for Lady-Bandit-Guard.

    The terminal events are mirrored; the shaping is not, because each side
    gets a potential over the distance it is trying to close. The shaping gives
    each side a dense gradient toward the right behavior without moving the
    game's equilibria, while the terminal events dominate the cumulative return
    when triggered. The potential is zero at an absorbing state, so the step
    that ends the episode pays no shaping beyond returning the potential the
    side had.

    ``shaping_scale_m`` is the distance that makes a potential unity,
    ``shaping_gain`` scales the shaping term (zero for a terminal-only game),
    ``shaping_discount`` must match the planner's per-step discount for a
    side's shaped value to be exactly its unshaped value less its own
    potential, and ``home_weight`` prices a guard's distance from the lady
    inside the guard's potential.

    ``catch_dwell_steps`` and ``breach_dwell_steps`` hold back the matching
    terminal bonus until a bandit has spent that many consecutive steps
    inside the radius, matching the termination condition so the bonus is
    paid on the step the episode ends. Zero pays it on the first step
    inside the radius.

    ``guard_separation_m``, ``lady_keepout_m`` and ``separation_cost`` set the
    guard-side crowding charge; ``dv_cost`` is the price of fuel in reward
    units per m/s of delta-v. Both are real per-side costs rather than
    shaping: each side pays only its own, so raising either makes that side
    more careful without handing the other an advantage.
    """

    shaping_scale_m: float = 300.0
    shaping_gain: float = 50.0
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
    catch_dwell_steps: int = 0
    breach_dwell_steps: int = 0
    escape_radius_m: float = 0.0
    repel_on_empty_tank: bool = False
    dv_cost: float = 0.0
    scope: RewardScope = RewardScope.PER_SIDE

    def __post_init__(self):
        if self.shaping_scale_m <= 0.0:
            raise ValueError(
                "shaping_scale_m divides every distance in the shaping potentials, "
                f"so it must be a positive length in metres; got {self.shaping_scale_m!r}. "
                "Set shaping_gain=0.0 to turn the shaping off instead."
            )

    def potential(self, state, side: Side) -> jax.Array:
        """:func:`lbg_potential` for ``side`` at this reward's scale and home weight."""
        return lbg_potential(state, side, self.shaping_scale_m, self.home_weight)

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

        caught, breached, _, _ = lbg_events_with_dwell(
            prev_state,
            next_state,
            params.dt,
            self.catch_radius_m,
            self.catch_speed_mps,
            self.breach_radius_m,
            self.breach_speed_mps,
            self.catch_dwell_steps,
            self.breach_dwell_steps,
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

        # The potential is zero at an absorbing state, so a terminating step
        # returns the whole of Phi(s) and the shaped return carries no residual
        # gain * g^T * Phi(s_T). That is the condition under which the
        # invariance is exact for an episodic game.
        terminating = jnp.logical_or(jnp.logical_or(caught, breached), repelled)
        terminating = jnp.logical_or(terminating, next_state.step >= params.max_steps)
        phi_next = jnp.where(terminating, 0.0, self.potential(next_state, side))
        shaping = self.shaping_gain * (
            self.shaping_discount * phi_next - self.potential(prev_state, side)
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
        return shaping + self.r_breach * breach_event - self.r_catch * guard_win - fuel
