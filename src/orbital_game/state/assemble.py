"""Assemble a per-scenario state class from a list of StateComponent types.

The resulting class is a flax.struct.dataclass (i.e., a JAX pytree, e.g.
GuardState/BanditState) with exactly the union of the components' fields, in a
deterministic order: components in the order passed, and each component's
fields in their insertion order.
"""

from __future__ import annotations

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
    """Return a new flax.struct.dataclass type combining all components' fields.

    The returned class exposes:
      - one attribute per field (with shape (n_vehicles, *field_shape))
      - a classmethod `zeros(n)` that constructs a zero-initialized instance
    """
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
    namespace["_orbital_game_components"] = tuple(components)
    namespace["_orbital_game_n_vehicles"] = n_vehicles

    # Create the class, then decorate it with flax.struct.dataclass.
    cls = type(class_name, (), namespace)
    return cast(type[Any], flax.struct.dataclass(cls))
