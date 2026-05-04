"""LeadInterceptPursuer — thrust toward opponent's predicted next-step position.

Heuristic-chaser pattern. Reads opponent state from a FullObservation-style
flat obs `[own_truth(6), opp_truth(6)]`, predicts opponent next-step under
zero-control HCW (free drift), and emits a max-magnitude impulse along the
line to that predicted point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class LeadInterceptPursuer:
    """Bandit (or guard) policy that closes on the opponent.

    Assumes the obs is a `FullObservation` flat layout in 1v1 RTN:
    `[own_rtn(6), opp_rtn(6)]`. For other observation channels, wrap in
    a `LeadInterceptPursuer`-shaped adapter that reconstructs the same
    own/opp split.
    """

    max_dv_mps: float = 0.05
    dt: float = 10.0
    n_vehicles: int = 0
    action_dim: int = 0

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, Any]:
        del key, t
        own = obs[0:6]
        opp = obs[6:12]
        opp_pos = opp[0:3]
        opp_vel = opp[3:6]
        # Free-drift one-step prediction (zero-control HCW limit).
        opp_pred = opp_pos + opp_vel * self.dt
        direction = opp_pred - own[0:3]
        unit = direction / (jnp.linalg.norm(direction) + 1e-9)
        dv = unit * self.max_dv_mps
        action = jnp.broadcast_to(dv, (self.n_vehicles, self.action_dim))
        return action, policy_state
