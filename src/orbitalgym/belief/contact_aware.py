"""ContactAwareBelief — duck-typed wrapper exposing `.mean` + `.contact`.

Policies that consume only `.mean` (existing MCTS, glideslope, lead-intercept,
etc.) are unaffected — the mean attribute is delegated. `PlanCachePolicy`
is the only consumer that reads `.contact`.
"""

from __future__ import annotations

from dataclasses import replace
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
    planning_context: PlanningContext | None = None

    @property
    def mean(self) -> jax.Array:
        return self.inner.mean


@flax.struct.dataclass
class PlanningContext:
    """Current own telemetry and observer-local estimated event history.

    Telemetry excludes every motion frame and all opposing-side data. A team
    container holds per-vehicle values; independent planners must select only
    the acting vehicle's slot. Counters are estimates, never hidden truth.
    """

    own_telemetry: dict[str, jax.Array]
    t: jax.Array
    step: jax.Array
    dwell_catch: jax.Array  # (N_self, N_bandits)
    dwell_breach: jax.Array


def initial_planning_context(env: Any, state: Any, side: Any) -> PlanningContext:
    """Read own non-motion telemetry and initialize unknown dwell to zero."""
    from dataclasses import fields

    import jax.numpy as jnp

    from orbitalgym.env.types import Side

    own = state.guards if side is Side.GUARD else state.bandits
    n_self = env.config.n_guards if side is Side.GUARD else env.config.n_bandits
    telemetry = {
        f.name: getattr(own, f.name) for f in fields(own) if f.name not in ("rt", "rtn", "eci")
    }
    counters = jnp.zeros((n_self, env.config.n_bandits), dtype=jnp.int32)
    return PlanningContext(telemetry, state.t, state.step, counters, counters)


def advance_planning_context(
    context: PlanningContext,
    previous_mean: jax.Array,
    next_mean: jax.Array,
    side: Any,
    env: Any,
    template_env_state: Any,
    next_state: Any,
) -> PlanningContext:
    """Advance each observer's estimated dwell using the game's segment rule.

    previous_mean is the local estimate used for the just-executed decision
    (after any declared fusion); next_mean is its updated posterior. Fusion
    does not copy or merge event histories. Current telemetry/time are read
    from next_state, whose opponent state and global counters are ignored.
    """
    import jax.numpy as jnp

    from orbitalgym.belief.flatten import belief_mean_to_env_state

    current = initial_planning_context(env, next_state, side)
    # Legacy flat/oracle and reduced-frame beliefs do not provide the local
    # truth-frame motion needed for segment bookkeeping. Their policies retain
    # their existing inputs; no estimated event history is invented here.
    if previous_mean.ndim != 3 or previous_mean.shape[-1] != 2 * env.truth_frame.dim:
        return current

    def advance(i):
        prev = belief_mean_to_env_state(
            previous_mean, side, env, template_env_state, observer_index=i, planning_context=context
        )
        nxt = belief_mean_to_env_state(
            next_mean, side, env, template_env_state, observer_index=i, planning_context=current
        )
        advanced = env.config.game.advance_state(prev, nxt, env.config)
        return advanced.dwell_catch, advanced.dwell_breach

    catch, breach = jax.vmap(advance)(jnp.arange(previous_mean.shape[0]))
    return replace(current, dwell_catch=catch, dwell_breach=breach)
