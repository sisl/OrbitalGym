"""Event-driven termination for the Lady-Bandit-Guard game.

Terminates when any of:
  - Max episode steps reached (read from ``params.max_steps`` at call time).
  - Any bandit is within ``breach_radius_m`` of the lady (origin) — bandit win.
  - Any guard is within ``catch_radius_m`` of any bandit — guard intercept.

The framing assumes the bandit is the threat trying to reach the lady and
the guard is defending. ``LadyBanditGuard.default_termination_fn`` wires
this onto a cfg automatically; constructing this class directly is also
supported via the ``termination_fn=`` constructor kwarg on ``ScenarioConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbitalgym.registry import TerminationFnKey, register


def _positions(side_state):
    if hasattr(side_state, "rtn"):
        return side_state.rtn[:, :3]
    return side_state.rt[:, :2]


@register(TerminationFnKey.LBG_EVENTS)
@dataclass(frozen=True)
class LbgEventTermination:
    """Terminate on max_steps OR bandit breaches lady OR guard catches bandit.

    The catch event is optional: set `catch_radius_m=0` to disable, leaving
    only the max-steps and breach gates. `max_steps` is read from
    `params.max_steps` at call time (single source of truth on the cfg).
    """

    breach_radius_m: float
    catch_radius_m: float = 0.0

    def __call__(self, state, params, t) -> jax.Array:
        del t
        hit_max = state.step >= params.max_steps

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
