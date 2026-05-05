"""BeliefConditionedPolicy — pre-pends a belief vector to the obs.

Useful for policies that should read belief mean (or full belief stats)
in addition to the raw observation. The caller threads the belief
through `policy_state` as a tuple `(belief_mean, base_state)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbital_game.policies.base import Policy


@dataclass(frozen=True)
class BeliefConditionedPolicy:
    """Wraps a base policy and prepends a belief vector to its observation.

    policy_state shape: `(belief_mean: jax.Array, base_state: Any)`. The
    belief vector is concatenated to the front of `obs` before delegating.
    """

    base: Policy
    n_vehicles: int = 0
    command_cls: Any = None

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        if self.command_cls is None:
            raise ValueError(
                "BeliefConditionedPolicy was called before the env injected "
                "`command_cls`. Use this policy via OrbitalGameEnv / "
                "SingleAgentView, or pass command_cls explicitly."
            )
        belief_mean, base_state = policy_state
        augmented_obs = jnp.concatenate([belief_mean, obs])
        cmd, next_base_state = self.base(base_state, augmented_obs, key, t)
        return cmd, (belief_mean, next_base_state)
