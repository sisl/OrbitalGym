"""Event-driven termination for the Lady-Bandit-Guard game.

Terminates when any of:
  - Max episode steps reached (read from ``params.max_steps`` at call time).
  - Any bandit passes within ``breach_radius_m`` of the lady (origin) during
    the step, slower than ``breach_speed_mps`` — bandit win.
  - Any guard passes within ``catch_radius_m`` of any bandit during the
    step, slower than ``catch_speed_mps`` — guard intercept.
  - Every bandit is repelled: driven beyond ``escape_radius_m``, or out of
    propellant on a coast that cannot reach the lady in the steps that
    remain — guard win without an intercept.

Both spatial events are evaluated over the whole step with
:func:`orbitalgym.games.proximity.lbg_events`, so an encounter peaking
between two decision samples still fires. The speed thresholds default to
infinity, which gates on radius alone.

The framing assumes the bandit is the threat trying to reach the lady and
the guard is defending. ``LadyBanditGuard.default_termination_fn`` wires
this onto a cfg automatically; constructing this class directly is also
supported via the ``termination_fn=`` constructor kwarg on ``ScenarioConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbitalgym.games.proximity import lbg_events, lbg_repelled
from orbitalgym.registry import TerminationFnKey, register


@register(TerminationFnKey.LBG_EVENTS)
@dataclass(frozen=True)
class LbgEventTermination:
    """Terminate on max_steps OR breach OR catch OR every bandit repelled.

    The catch event is optional: set `catch_radius_m=0` to disable, leaving
    only the max-steps and breach gates. Both repel gates are off by
    default: `escape_radius_m=0` disables the distance gate and
    `repel_on_empty_tank=False` the propellant gate. `max_steps`, the step
    duration and the reference orbit are read from `params` at call time
    (single source of truth on the cfg).
    """

    breach_radius_m: float
    catch_radius_m: float = 0.0
    breach_speed_mps: float = float("inf")
    catch_speed_mps: float = float("inf")
    escape_radius_m: float = 0.0
    repel_on_empty_tank: bool = False

    def __call__(self, prev_state, state, params, t) -> jax.Array:
        del t
        hit_max = state.step >= params.max_steps
        caught, breached, _, _ = lbg_events(
            prev_state,
            state,
            params.dt,
            self.catch_radius_m,
            self.catch_speed_mps,
            self.breach_radius_m,
            self.breach_speed_mps,
        )
        repelled = lbg_repelled(
            state,
            params,
            self.escape_radius_m,
            self.repel_on_empty_tank,
            self.breach_radius_m,
            self.breach_speed_mps,
        )
        return jnp.logical_or(hit_max, jnp.logical_or(breached, jnp.logical_or(caught, repelled)))
