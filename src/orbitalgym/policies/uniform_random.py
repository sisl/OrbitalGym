"""UniformRandomDiscretePolicy — uniform sample from a discrete action grid.

Default ``opponent_model`` for :class:`orbitalgym.policies.mcts.MCTSPolicy`.
Each vehicle on the side independently samples an action index uniformly
from the rows of ``action_grid``; the resulting Δv vectors are padded to
the side's command ``dv_dim`` and emitted as a Command.

This is *not* a competitive policy in the real game — it is a deliberately
cheap default for the searcher's *belief about the opponent*. Real-game
play should wire a stronger Policy (e.g. ``LQRBanditPolicy`` or
``LeadInterceptPursuer``) on the opposing side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class UniformRandomDiscretePolicy:
    """Uniform-random over a (A, dv_dim) discrete action grid, per vehicle.

    Env-populated init knobs (filled by OrbitalGymEnv via
    ``dataclasses.replace``):
        command_cls: per-side Command class
        n_vehicles: number of vehicles on this side

    The ``action_grid`` row count is the discrete action-space size; rows
    are Δv vectors whose dimensionality is padded to the side's command
    ``dv_dim`` if needed (e.g. 2-D RT grid into a 3-D RTN command).
    """

    action_grid: jax.Array  # (A, dv_dim)
    n_vehicles: int = 0
    command_cls: Any = None

    def __call__(
        self,
        policy_state: Any,
        agent_view: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        del agent_view, t
        if self.command_cls is None:
            raise ValueError(
                "UniformRandomDiscretePolicy was called before the env injected "
                "`command_cls`. Pass command_cls=... and n_vehicles=... explicitly."
            )

        n = self.n_vehicles
        a = self.action_grid.shape[0]
        keys = jax.random.split(key, n)
        idxs = jax.vmap(lambda k: jax.random.randint(k, (), 0, a))(keys)
        dvs = self.action_grid[idxs]  # (n, dv_dim_grid)

        cmd_template = self.command_cls.zeros(n)
        target_dim = cmd_template.dv.shape[-1]
        grid_dim = dvs.shape[-1]
        if grid_dim < target_dim:
            pad = jnp.zeros((n, target_dim - grid_dim), dvs.dtype)
            dvs = jnp.concatenate([dvs, pad], axis=-1)
        elif grid_dim > target_dim:
            dvs = dvs[..., :target_dim]
        return cmd_template.replace(dv=dvs.astype(cmd_template.dv.dtype)), policy_state
