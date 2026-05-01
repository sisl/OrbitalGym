"""StateLayout — the bridge between split (guard, bandit) pytrees and the
single flat vector view that POMDP planners consume.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
from jax.flatten_util import ravel_pytree


@dataclass(frozen=True)
class StateLayout:
    guard_state_cls: type
    bandit_state_cls: type
    n_guards: int
    n_bandits: int
    flat_dim: int
    _flatten_fn: Callable = field(repr=False)
    _unflatten_fn: Callable = field(repr=False)

    @classmethod
    def build(
        cls,
        guard_state_cls: type,
        bandit_state_cls: type,
        n_guards: int,
        n_bandits: int,
    ) -> StateLayout:
        """Construct a StateLayout using zero-initialized states to lock the
        pytree treedef. The treedef is what makes the flat ordering deterministic."""
        zero_guard = guard_state_cls.zeros(n_guards)
        zero_bandit = bandit_state_cls.zeros(n_bandits)
        zero_vec, unravel = ravel_pytree((zero_guard, zero_bandit))

        def _flatten(guards, bandits):
            vec, _ = ravel_pytree((guards, bandits))
            return vec

        def _unflatten(vec):
            return unravel(vec)

        return cls(
            guard_state_cls=guard_state_cls,
            bandit_state_cls=bandit_state_cls,
            n_guards=n_guards,
            n_bandits=n_bandits,
            flat_dim=int(zero_vec.size),
            _flatten_fn=_flatten,
            _unflatten_fn=_unflatten,
        )

    # ---- per-vehicle accessors ----

    def guard(self, guard_state: Any, i: int) -> Any:
        """Return the ith guard's state as a per-vehicle pytree
        (same fields, leading axis indexed away)."""
        return jax.tree_util.tree_map(lambda x: x[i], guard_state)

    def bandit(self, bandit_state: Any, j: int) -> Any:
        """Return the jth bandit's state as a per-vehicle pytree."""
        return jax.tree_util.tree_map(lambda x: x[j], bandit_state)

    # ---- flat view (for POMDP planners) ----

    def flatten(self, guard_state: Any, bandit_state: Any) -> jax.Array:
        return self._flatten_fn(guard_state, bandit_state)

    def unflatten(self, vec: jax.Array) -> tuple[Any, Any]:
        return self._unflatten_fn(vec)
