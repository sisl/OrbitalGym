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

    Only the `dv` field is gated by confidence. Non-`dv` Command fields
    (e.g. `Communicate.active`, `Communicate.payload`) propagate from the
    **primary** policy unchanged — the fallback's non-`dv` fields are
    discarded. If the fallback should drive non-`dv` fields too, replace
    this wrapper with a custom routing policy.
    """

    primary: Policy
    fallback: Policy
    threshold: float = 0.5
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
                "HeuristicWithFallbackPolicy was called before the env injected "
                "`command_cls`. Use this policy via OrbitalGameEnv / "
                "SingleAgentView, or pass command_cls explicitly."
            )
        confidence = jnp.asarray(policy_state)
        k1, k2 = jax.random.split(key)
        primary_cmd, _ = self.primary(None, obs, k1, t)
        fallback_cmd, _ = self.fallback(None, obs, k2, t)
        chosen_dv = jnp.where(confidence >= self.threshold, primary_cmd.dv, fallback_cmd.dv)
        # Take non-dv fields from the primary so e.g. Communicate.active is preserved.
        return primary_cmd.replace(dv=chosen_dv), policy_state
