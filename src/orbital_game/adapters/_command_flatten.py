"""Shared Command-pytree <-> flat-array helpers used by adapter modules.

Keeps the adapter-specific code lean: each adapter just calls
`command_flat_dim(cmd_cls)` to size its action space, and
`flatten_command(cmd)` / `unflatten_command(cmd_cls, flat)` to round-trip.

Field order is registration order: components in the order declared on the
Command class, and each component's fields in their `fields()` insertion order.
This matches the assembly contract in `actions/assemble.py`, so the flat layout
is stable across builds of the same `(components, n_agents)` tuple.
"""

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp


def command_flat_dim(command_cls) -> int:
    """Total flat dim of all component fields concatenated."""
    n = command_cls._orbital_game_n_agents
    total = 0
    for comp in command_cls._orbital_game_action_components:
        for _, fshape in comp.fields().items():
            per_agent = prod(fshape) if fshape else 1
            total += n * per_agent
    return total


def flatten_command(cmd) -> jax.Array:
    """Concatenate every field in registration order, flattened.

    Returns a zero-length float array when the Command has no fields
    (empty action-component tuple), which is valid for dynamics-only
    scenarios where env.step propagates translation without any action.
    """
    parts = []
    for comp in type(cmd)._orbital_game_action_components:
        for fname in comp.fields():
            v = getattr(cmd, fname)
            parts.append(v.reshape(-1))
    if not parts:
        return jnp.zeros((0,))
    return jnp.concatenate(parts)


def unflatten_command(command_cls, flat: jax.Array):
    """Inverse of flatten_command. Driven off field shapes.

    Flat-array transports lose dtype (heterogeneous fields are upcast to the
    common dtype during concatenate, typically float). We restore each
    field's declared dtype by reading it from the component's `zeros(1)`
    template before reshaping. This keeps boolean/integer fields like
    `Communicate.active` round-tripping through Gym/PettingZoo/POMDP
    adapters without silent corruption.
    """
    n = command_cls._orbital_game_n_agents
    vals: dict = {}
    cursor = 0
    for comp in command_cls._orbital_game_action_components:
        # One template per component is enough — dtype is shape-independent.
        template = comp.zeros(1)
        for fname, fshape in comp.fields().items():
            per_agent = prod(fshape) if fshape else 1
            size = n * per_agent
            slice_ = flat[cursor : cursor + size].astype(template[fname].dtype)
            vals[fname] = slice_.reshape((n, *fshape)) if fshape else slice_.reshape((n,))
            cursor += size
    return command_cls(**vals)
