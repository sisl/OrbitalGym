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

    Env-populated knobs (filled by OrbitalGameEnv at construction):
        n_vehicles: number of vehicles on this side
        command_cls: per-side Command pytree class
    """

    max_dv_mps: float = 0.05
    dt: float = 10.0

    n_vehicles: int = 0
    command_cls: Any = None

    def __call__(
        self,
        policy_state: Any,
        agent_view: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        del key, t
        # FullObservation flattens the per-(observer, tracked) tensor as
        # [own_truth, opp_truth] — own side first, then opposing side.
        # For a 1v1 bandit observer the layout is [bandit_rtn(6), guard_rtn(6)].
        bandit_rtn = agent_view[0:6]
        guard_rtn = agent_view[6:12]

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

        # dv shape (n_vehicles, 3) — single bandit, RTN.
        dv_per_vehicle = jnp.broadcast_to(dv, (self.n_vehicles, 3))
        cmd = self.command_cls.zeros(self.n_vehicles).replace(dv=dv_per_vehicle)
        return cmd, policy_state
