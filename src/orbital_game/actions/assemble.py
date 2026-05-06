"""Assemble a per-scenario per-side Command class from ActionComponent instances.

The resulting class is a flax.struct.dataclass (a JAX pytree) with exactly the
union of the components' fields, in registration order. Mirrors
`state.assemble.build_state_class` byte-for-byte; same caching contract for
`jax.lax.while_loop` carry-pytree-structure equality.

Components are passed as *instances* (not classes) so frame-aware components
like ImpulsiveManeuver can report the correct field shape based on their
configured ``action_frame``.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, cast

import flax.struct
import jax

from orbital_game.actions.components import ActionComponent


def build_command_class(
    components: Sequence[ActionComponent],
    n_agents: int,
    class_name: str,
) -> type[Any]:
    """Return a flax.struct.dataclass type combining all components' fields.

    The returned class exposes:
      - one attribute per field (with shape (n_agents, *field_shape))
      - a classmethod `zeros(n)` that constructs an identity-command instance
    """
    return _build_command_class_cached(tuple(components), n_agents, class_name)


@functools.cache
def _build_command_class_cached(
    components: tuple[ActionComponent, ...],
    n_agents: int,
    class_name: str,
) -> type[Any]:
    field_specs: list[tuple[str, tuple[int, ...]]] = []
    for comp in components:
        for fname, fshape in comp.fields().items():
            field_specs.append((fname, fshape))

    annotations = {fname: jax.Array for fname, _ in field_specs}
    namespace: dict = {"__annotations__": annotations}

    def _zeros(cls, n: int):
        vals: dict = {}
        for comp in components:
            vals.update(comp.zeros(n))
        return cls(**vals)

    namespace["zeros"] = classmethod(_zeros)
    namespace["_orbital_game_action_components"] = components
    namespace["_orbital_game_n_agents"] = n_agents

    cls = type(class_name, (), namespace)
    return cast(type[Any], flax.struct.dataclass(cls))
