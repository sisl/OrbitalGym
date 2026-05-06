"""Event-driven termination for the Lady-Bandit-Guard game.

Terminates when any of:
  - Max episode steps reached.
  - Any bandit is within `breach_radius_m` of the lady (origin) — bandit win.
  - Any guard is within `catch_radius_m` of any bandit — guard intercept.

Differs from `MaxStepsOrBreach`, which checks the guard's distance to the
origin. That termination is appropriate when the guard *is* the threat (the
original LBG framing); for the dog-food scenario the bandit is the threat
and the guard is defending, so the breach must trigger on bandit→lady.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _positions(side_state):
    if hasattr(side_state, "rtn"):
        return side_state.rtn[:, :3]
    return side_state.rt[:, :2]


@dataclass(frozen=True)
class LbgEventTermination:
    """Terminate on max_steps OR bandit breaches lady OR guard catches bandit.

    The catch event is optional: set `catch_radius_m=0` to disable, leaving
    only the max-steps and breach gates.
    """

    max_steps: int
    breach_radius_m: float
    catch_radius_m: float = 0.0

    def __call__(self, state, params, t) -> jax.Array:
        del params, t
        hit_max = state.step >= self.max_steps

        bandit_pos = _positions(state.bandits)
        d_bandit_lady_min = jnp.min(jnp.linalg.norm(bandit_pos, axis=-1))
        breached = d_bandit_lady_min < self.breach_radius_m

        if self.catch_radius_m > 0.0:
            guard_pos = _positions(state.guards)
            diffs = guard_pos[:, None, :] - bandit_pos[None, :, :]
            d_gb_min = jnp.min(jnp.linalg.norm(diffs, axis=-1))
            caught = d_gb_min < self.catch_radius_m
        else:
            caught = jnp.asarray(False)

        return jnp.logical_or(hit_max, jnp.logical_or(breached, caught))
