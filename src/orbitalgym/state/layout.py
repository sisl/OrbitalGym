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
    """Bridge between the per-side pytree state and a single flat vector.

    Bundles four things that derive together from `n_guards`,
    `n_bandits`, and the per-side state-component tuples
    (`cfg.guard_components`, `cfg.bandit_components`):

    - `guard_state_cls`, `bandit_state_cls` — the dynamically-assembled
      per-side dataclass types. These contain one field per state
      component (e.g. `rtn` for an RTN dynamics frame, `mass`,
      `power`, ...). The set of fields is determined by the
      `cfg.{side}_components` tuple.
    - `n_guards`, `n_bandits` — vehicle counts.
    - `flat_dim` — the size of the full flat state vector. For a 1v1
      RTN scenario with masses, this is `(6 + 1) + (6 + 1) = 14`
      (six dims of state per vehicle from RTN, one extra from MASS,
      times two sides). Custom component sets change this.
    - `_flatten_fn`, `_unflatten_fn` — JIT-compatible bijections
      between `(guards_pytree, bandits_pytree)` and a flat
      `(flat_dim,)` vector. POMDP planners use these to operate on
      the flat representation.

    Built once per cfg in `ScenarioConfig.__post_init__`. Read it as
    `cfg.layout`.
    """

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

    # ---- dynamics state dim (for belief / observation construction) ----

    @property
    def dynamics_state_dim(self) -> int:
        """Per-vehicle dynamics state dim — read from the assembled state class."""
        from orbitalgym.state.components import RTNState, RTState

        comps = self.guard_state_cls._orbitalgym_components
        if RTNState in comps:
            return 6
        if RTState in comps:
            return 4
        raise AttributeError("State has no dynamics component (RTState or RTNState)")
