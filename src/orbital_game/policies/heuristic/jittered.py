"""JitteredPolicy — wraps a base Policy and adds PRNG-keyed Gaussian jitter.

Useful for randomized heuristic policies: take a deterministic policy
(e.g. `LeadInterceptPursuer`) and add stochastic perturbations on top.

`policy_state` is forwarded to the base policy unchanged. The wrapper
itself has no internal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from orbital_game.policies.base import Policy


@dataclass(frozen=True)
class JitteredPolicy:
    """Wraps a base policy and adds PRNG-keyed Gaussian jitter to its action.

    The jitter is drawn from N(0, sigma²I) with the same shape as the
    base action. Setting `sigma=0.0` is a pure passthrough.

    `base` is required — instantiate with `JitteredPolicy(base=..., sigma=...)`.
    """

    base: Policy
    sigma: float = 0.01
    n_vehicles: int = 0
    action_dim: int = 0

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, Any]:
        k1, k2 = jax.random.split(key)
        base_action, next_state = self.base(policy_state, obs, k1, t)
        noise = self.sigma * jax.random.normal(k2, shape=base_action.shape)
        return base_action + noise, next_state
