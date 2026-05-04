"""OrthogonalEvader — move perpendicular to opponent relative velocity.

Picks a unit direction in the plane orthogonal to the opponent's relative
velocity vector. With zero relative velocity the orthogonal subspace is
ill-defined, so we fall back to a fixed reference direction (+y, the
along-track axis in RTN) to keep the action finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

_FALLBACK_AXIS = jnp.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class OrthogonalEvader:
    """Evades an approaching opponent by thrusting perpendicular to relative velocity.

    Assumes obs layout `[own_rtn(6), opp_rtn(6)]`. Action lies in the
    orthogonal complement of the relative velocity vector.
    """

    max_dv_mps: float = 0.05
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
        rel_v = opp[3:6] - own[3:6]
        norm = jnp.linalg.norm(rel_v)
        # Pick a reference axis that's not parallel to rel_v.
        # Project _FALLBACK_AXIS onto orthogonal complement of rel_v.
        unit_v = jnp.where(norm > 1e-9, rel_v / (norm + 1e-12), _FALLBACK_AXIS)
        proj = _FALLBACK_AXIS - jnp.dot(_FALLBACK_AXIS, unit_v) * unit_v
        proj_norm = jnp.linalg.norm(proj)
        # If _FALLBACK_AXIS is collinear with rel_v, use the radial axis instead.
        backup = jnp.array([1.0, 0.0, 0.0])
        backup_proj = backup - jnp.dot(backup, unit_v) * unit_v
        ortho = jnp.where(proj_norm > 1e-6, proj, backup_proj)
        ortho_unit = ortho / (jnp.linalg.norm(ortho) + 1e-12)
        dv = ortho_unit * self.max_dv_mps
        action = jnp.broadcast_to(dv, (self.n_vehicles, self.action_dim))
        return action, policy_state
