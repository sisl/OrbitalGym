"""SunTrackerBlocker — move to maintain Sun-line geometry vs the opponent.

Sun-line blocking pattern: the policy knows a Sun-direction unit vector in
the local RTN frame and thrusts along the line from the controlled vehicle
toward the projection of the opponent onto the Sun line. The Sun direction
is supplied as a static knob (`sun_dir_rtn`) so the policy stays self-
contained and doesn't reach into cfg/params.

Real SunBlocking deployments would re-derive the Sun direction per step
from epoch + time (see `games/sun_blocking.py:_sun_eci_at_mjd`); this
gallery class is the static-Sun teaching version. A dynamic-Sun variant is
out of scope for this gallery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp


def _default_sun_dir() -> jax.Array:
    return jnp.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class SunTrackerBlocker:
    """Sun-line-aware heuristic blocker.

    Assumes obs layout `[own_rtn(6), opp_rtn(6)]`. Thrusts along the
    Sun-direction line, oriented by the sign of the opponent's projection
    onto that line (so the thrust closes on the Sun-side of the opponent).
    """

    sun_dir_rtn: jax.Array = field(default_factory=_default_sun_dir)
    max_dv_mps: float = 0.05
    n_vehicles: int = 0
    command_cls: Any = None

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        del key, t
        if self.command_cls is None:
            raise ValueError(
                "SunTrackerBlocker was called before the env injected `command_cls`. "
                "Use this policy via OrbitalGameEnv / SingleAgentView, or pass "
                "command_cls explicitly."
            )
        own = obs[0:6]
        opp = obs[6:12]
        rel = opp[0:3] - own[0:3]
        sun_unit = self.sun_dir_rtn / (jnp.linalg.norm(self.sun_dir_rtn) + 1e-12)
        # Project rel onto the Sun line; thrust along that projection's sign.
        scalar = jnp.dot(rel, sun_unit)
        # If projection is zero, fall back to the Sun direction itself.
        sign = jnp.where(jnp.abs(scalar) > 1e-9, jnp.sign(scalar), 1.0)
        direction = sun_unit * sign
        dv = direction * self.max_dv_mps
        cmd_template = self.command_cls.zeros(self.n_vehicles)
        dv_dim = cmd_template.dv.shape[-1]
        dv_per_vehicle = jnp.broadcast_to(dv[:dv_dim], (self.n_vehicles, dv_dim))
        return cmd_template.replace(dv=dv_per_vehicle), policy_state
