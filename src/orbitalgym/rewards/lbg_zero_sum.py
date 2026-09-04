"""Zero-sum LBG reward with dense distance shaping and terminal events.

Both sides receive a per-step signal:

    guard:
        -alpha * d_guard_bandit_min                  (dense shaping)
        + R_catch  * 1[catch event]                  (caught bandit terminal bonus)
        - R_breach * 1[breach event]                 (lady breached terminal penalty)

    bandit (mirror):
        -alpha * d_bandit_lady_min                   (dense shaping)
        + R_breach * 1[breach event]                 (intercepted lady terminal bonus)
        - R_catch  * 1[catch event]                  (caught by guard terminal penalty)

Distances and events come from
:func:`orbitalgym.games.proximity.lbg_events` over the step from
``prev_state`` to ``next_state``, the same source
:class:`~orbitalgym.termination.lbg_events.LbgEventTermination` reads, so a
terminal bonus is paid in exactly the step the episode ends:

    d_guard_bandit_min = min over (g, b) of the guard-bandit closest approach
    d_bandit_lady_min  = min over b      of the bandit-lady closest approach

Lady is virtual — fixed at the RTN origin (the reference orbit). The reward is
scope=PER_SIDE: returns a scalar per side and is JIT-friendly (no Python branches
on traced values).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from orbitalgym.env.types import Side
from orbitalgym.games.proximity import lbg_events
from orbitalgym.registry import RewardFnKey, register
from orbitalgym.rewards.base import RewardScope


@register(RewardFnKey.LBG_ZERO_SUM)
@dataclass(frozen=True)
class LbgZeroSumReward:
    """Dense + terminal zero-sum reward for the Lady-Bandit-Guard game.

    The two sides receive mirrored signals. With the default weights, the
    dense term provides a steady gradient toward the right behavior while
    the terminal events dominate the cumulative return when triggered.
    """

    alpha: float = 1e-3
    r_catch: float = 1000.0
    r_breach: float = 1000.0
    catch_radius_m: float = 50.0
    breach_radius_m: float = 5.0
    catch_speed_mps: float = float("inf")
    breach_speed_mps: float = float("inf")
    scope: RewardScope = RewardScope.PER_SIDE

    def __call__(self, prev_state, action, next_state, side, params, t):
        del action, t

        caught, breached, d_gb_min, d_bl_min = lbg_events(
            prev_state,
            next_state,
            params.dt,
            self.catch_radius_m,
            self.catch_speed_mps,
            self.breach_radius_m,
            self.breach_speed_mps,
        )
        catch_event = caught.astype(jnp.float32)
        breach_event = breached.astype(jnp.float32)

        if side is Side.GUARD:
            return (
                -self.alpha * d_gb_min + self.r_catch * catch_event - self.r_breach * breach_event
            )
        return -self.alpha * d_bl_min + self.r_breach * breach_event - self.r_catch * catch_event
