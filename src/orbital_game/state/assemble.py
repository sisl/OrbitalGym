"""Assemble a per-scenario state class from a list of StateComponent types.

The resulting class is a flax.struct.dataclass (i.e., a JAX pytree, e.g.
GuardState/BanditState) with exactly the union of the components' fields, in a
deterministic order: components in the order passed, and each component's
fields in their insertion order.

build_state_class is cached on (components, n_vehicles, class_name) — JAX's
`jax.lax.while_loop` checks carry-pytree structure by Python-class identity,
so two same-shaped-but-distinct classes raise a TypeError. Caching guarantees
that any caller asking for the same shape gets the same class object.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, cast

import flax.struct
import jax

from orbital_game.state.components import StateComponent


def build_state_class(
    components: Sequence[type[StateComponent]],
    n_vehicles: int,
    class_name: str,
) -> type[Any]:
    """Return a flax.struct.dataclass type combining all components' fields.

    The returned class exposes:
      - one attribute per field (with shape (n_vehicles, *field_shape))
      - a classmethod `zeros(n)` that constructs a zero-initialized instance

    Repeated calls with the same `(components, n_vehicles, class_name)` return
    the same class object (required for `jax.lax.while_loop` carry equality).
    """
    return _build_state_class_cached(tuple(components), n_vehicles, class_name)


@functools.cache
def _build_state_class_cached(
    components: tuple[type[StateComponent], ...],
    n_vehicles: int,
    class_name: str,
) -> type[Any]:
    # Collect (field_name, shape) in deterministic order.
    field_specs: list[tuple[str, tuple[int, ...]]] = []
    for comp in components:
        for fname, fshape in comp.fields().items():
            field_specs.append((fname, fshape))

    # Build the annotations dict for the flax dataclass.
    annotations = {fname: jax.Array for fname, _ in field_specs}

    namespace: dict = {
        "__annotations__": annotations,
    }

    def _zeros(cls, n: int):
        vals: dict = {}
        for comp in components:
            vals.update(comp.zeros(n))
        return cls(**vals)

    namespace["zeros"] = classmethod(_zeros)
    namespace["_orbital_game_components"] = components
    namespace["_orbital_game_n_vehicles"] = n_vehicles

    # Create the class, then decorate it with flax.struct.dataclass.
    cls = type(class_name, (), namespace)
    return cast(type[Any], flax.struct.dataclass(cls))
