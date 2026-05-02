"""LeadInterceptPursuer — active pursuit via HCW one-step prediction.

At each step, predicts the guard's RTN position one step ahead under
HCW dynamics with zero control, and commands a max-magnitude impulse
toward that predicted point. Single-vehicle bandits only.

This is example code, not library code — lives under examples/ rather
than src/. The Tutorial T3 walkthrough builds this from scratch and
references this file as the worked solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class LeadInterceptPursuer:
    """Bandit policy that thrusts toward the guard's predicted next-step position.

    Args:
        max_dv_mps: Magnitude of the impulse command, m/s.
        dt: Step size in seconds (used for one-step HCW propagation).

    Env-populated dims (filled by OrbitalGameEnv at construction):
        n_vehicles, action_dim
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
        # FullObservation flattens the per-(observer, tracked) tensor as
        # [own_truth, opp_truth] — own side first, then opposing side.
        # For a 1v1 bandit observer the layout is [bandit_rtn(6), guard_rtn(6)].
        bandit_rtn = obs[0:6]
        guard_rtn = obs[6:12]

        # One-step HCW propagation under zero control: r' = r + v*dt.
        # Crude but adequate for one-step lead.
        guard_pos = guard_rtn[0:3]
        guard_vel = guard_rtn[3:6]
        guard_pred = guard_pos + guard_vel * self.dt

        bandit_pos = bandit_rtn[0:3]
        direction = guard_pred - bandit_pos
        norm = jnp.linalg.norm(direction) + 1e-9
        unit = direction / norm
        dv = unit * self.max_dv_mps

        # Action shape (n_vehicles, action_dim) — single bandit, RTN.
        action = jnp.broadcast_to(dv, (self.n_vehicles, self.action_dim))
        return action, policy_state
