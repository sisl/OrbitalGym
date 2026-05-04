"""HeuristicWithFallbackPolicy — confidence-gated routing between primary
and fallback policies.

The confidence signal is threaded through `policy_state`. Concretely,
`policy_state` is expected to be a scalar `jax.Array` in [0, 1]; values
below `threshold` route to `fallback`. The caller is responsible for
producing the confidence signal — typically from belief covariance or a
critic head.

This pattern is for hybrid systems where a learned policy has occasional
high-uncertainty regions and you want a known-safe fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbital_game.policies.base import Policy


@dataclass(frozen=True)
class HeuristicWithFallbackPolicy:
    """Confidence-gated routing wrapper.

    policy_state semantics: a scalar confidence in [0, 1]. The wrapped
    primary's own state is not threaded through this class — primaries
    that need state should be wrapped with their own state-handling.
    """

    primary: Policy
    fallback: Policy
    threshold: float = 0.5
    n_vehicles: int = 0
    action_dim: int = 0

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, Any]:
        confidence = jnp.asarray(policy_state)
        k1, k2 = jax.random.split(key)
        primary_action, _ = self.primary(None, obs, k1, t)
        fallback_action, _ = self.fallback(None, obs, k2, t)
        action = jnp.where(confidence >= self.threshold, primary_action, fallback_action)
        return action, policy_state
