"""Planner protocol — full-state planners that don't fit the Policy contract.

Policies (`policies.base.Policy`) are pure callables `(policy_state, obs, key,
t) -> (command, ps)`. They run inside `jax.lax.scan`, so they must be
JAX-traceable: no Python control flow over traced values, no Python-level
mutable tree state. Tree-search planners (MCTS, POMCPOW) violate both — the
tree expansion has variable structure and the search loop maintains Python
state across iterations.

`Planner` is the sibling protocol for these. It receives the full `EnvState`
(not a partial observation), runs a Python-level planning step, and returns
a Command pytree for its side. Driven by `rollout_with_planner` (a Python
loop over jit'd env steps) rather than the lax.scan-based `rollout`.

Conversion to a Policy via `jax.pure_callback` is possible but loses the
batched-rollout benefit; in practice, planners are best run from a manual
outer loop.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax


@runtime_checkable
class Planner(Protocol):
    """Structural protocol for a state-aware planner.

    Implementations must expose a ``plan(env_state, key) -> (command, planner_state)``
    method. The first return is the per-side Command pytree; the second is
    arbitrary Python state (e.g. a tree of node visits) that the caller may
    thread between calls. Stateless planners can return `None`.

    The Planner sees the full `EnvState`, including reference orbit and time.
    Implementations are typically frozen dataclasses with cached jitted
    rollouts and a Python-level outer loop for tree expansion / UCB.
    """

    def plan(
        self,
        env_state: Any,  # EnvState — kept Any to avoid circular import
        key: jax.Array,
    ) -> tuple[Any, Any]: ...
