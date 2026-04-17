"""StateLayout — the bridge between split (defender, intruder) pytrees and the
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
    defender_state_cls: type
    intruder_state_cls: type
    n_defenders: int
    n_intruders: int
    flat_dim: int
    _flatten_fn: Callable = field(repr=False)
    _unflatten_fn: Callable = field(repr=False)

    @classmethod
    def build(
        cls,
        defender_state_cls: type,
        intruder_state_cls: type,
        n_defenders: int,
        n_intruders: int,
    ) -> StateLayout:
        """Construct a StateLayout using zero-initialized states to lock the
        pytree treedef. The treedef is what makes the flat ordering deterministic."""
        zero_def = defender_state_cls.zeros(n_defenders)
        zero_int = intruder_state_cls.zeros(n_intruders)
        zero_vec, unravel = ravel_pytree((zero_def, zero_int))

        def _flatten(defs, ints):
            vec, _ = ravel_pytree((defs, ints))
            return vec

        def _unflatten(vec):
            return unravel(vec)

        return cls(
            defender_state_cls=defender_state_cls,
            intruder_state_cls=intruder_state_cls,
            n_defenders=n_defenders,
            n_intruders=n_intruders,
            flat_dim=int(zero_vec.size),
            _flatten_fn=_flatten,
            _unflatten_fn=_unflatten,
        )

    # ---- per-vehicle accessors ----

    def defender(self, defender_state: Any, i: int) -> Any:
        """Return the ith defender's state as a per-vehicle pytree
        (same fields, leading axis indexed away)."""
        return jax.tree_util.tree_map(lambda x: x[i], defender_state)

    def intruder(self, intruder_state: Any, j: int) -> Any:
        """Return the jth intruder's state as a per-vehicle pytree."""
        return jax.tree_util.tree_map(lambda x: x[j], intruder_state)

    # ---- flat view (for POMDP planners) ----

    def flatten(self, defender_state: Any, intruder_state: Any) -> jax.Array:
        return self._flatten_fn(defender_state, intruder_state)

    def unflatten(self, vec: jax.Array) -> tuple[Any, Any]:
        return self._unflatten_fn(vec)
