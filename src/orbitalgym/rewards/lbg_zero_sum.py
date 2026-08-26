"""Zero-sum LBG reward with dense distance shaping and terminal events.

Both sides receive a per-step signal:

    guard:
        -alpha * d_guard_bandit_min                  (dense shaping)
        + R_catch  * 1[d_guard_bandit_min < r_catch]   (caught bandit terminal bonus)
        - R_breach * 1[d_bandit_lady_min < r_breach]   (lady breached terminal penalty)

    bandit (mirror):
        -alpha * d_bandit_lady_min                   (dense shaping)
        + R_breach * 1[d_bandit_lady_min < r_breach] (intercepted lady terminal bonus)
        - R_catch  * 1[d_guard_bandit_min < r_catch]  (caught by guard terminal penalty)

Where:
    d_guard_bandit_min = min over (g, b) of ||guard_pos[g] - bandit_pos[b]||
    d_bandit_lady_min  = min over b      of ||bandit_pos[b]||           (lady at origin)

Lady is virtual — fixed at the RTN origin (the reference orbit). The reward is
scope=PER_SIDE: returns a scalar per side and is JIT-friendly (no Python branches
on traced values).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from orbitalgym.env.types import Side
from orbitalgym.registry import RewardFnKey, register
from orbitalgym.rewards.base import RewardScope


def _positions(side_state):
    """Extract per-vehicle position vectors from a side state pytree.

    Falls back from RTN to RT, mirroring DistanceToReferenceOrbit.
    """
    if hasattr(side_state, "rtn"):
        return side_state.rtn[:, :3]
    return side_state.rt[:, :2]


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
    scope: RewardScope = RewardScope.PER_SIDE

    def __call__(self, prev_state, action, next_state, side, params, t):
        del prev_state, action, params, t

        guard_pos = _positions(next_state.guards)  # (n_g, dim)
        bandit_pos = _positions(next_state.bandits)  # (n_b, dim)

        # Pairwise guard-bandit distances: (n_g, n_b).
        diffs = guard_pos[:, None, :] - bandit_pos[None, :, :]
        d_gb = jnp.linalg.norm(diffs, axis=-1)
        d_gb_min = jnp.min(d_gb)

        # Bandit-to-lady distances (lady at origin): (n_b,).
        d_bl = jnp.linalg.norm(bandit_pos, axis=-1)
        d_bl_min = jnp.min(d_bl)

        catch_event = (d_gb_min < self.catch_radius_m).astype(jnp.float32)
        breach_event = (d_bl_min < self.breach_radius_m).astype(jnp.float32)

        if side is Side.GUARD:
            return (
                -self.alpha * d_gb_min + self.r_catch * catch_event - self.r_breach * breach_event
            )
        return -self.alpha * d_bl_min + self.r_breach * breach_event - self.r_catch * catch_event
