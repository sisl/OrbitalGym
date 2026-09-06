"""ContactAwareBelief — duck-typed wrapper exposing `.mean` + `.contact`.

Policies that consume only `.mean` (existing MCTS, glideslope, lead-intercept,
etc.) are unaffected — the mean attribute is delegated. `PlanCachePolicy`
is the only consumer that reads `.contact`.
"""

from __future__ import annotations

from typing import Any

import flax.struct
import jax


@flax.struct.dataclass
class ContactAwareBelief:
    """Wraps an existing Belief with a per-agent contact mask.

    The wrapped belief is stored as a pytree node so jax.lax.scan threads
    it transparently. `.mean` is a property delegating to the inner belief
    (so `agent_view.mean` stays the canonical access pattern).
    """

    inner: Any
    contact: jax.Array  # (N_self,) bool

    @property
    def mean(self) -> jax.Array:
        return self.inner.mean
