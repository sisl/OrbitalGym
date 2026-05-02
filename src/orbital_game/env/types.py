"""Symmetric-core data types for OrbitalGameEnv.

`Side` is a Python-level discriminator (StrEnum, never a traced array).
`BySide[T]` is the universal one-per-side container — itself a flax pytree,
so `jax.vmap`/`jax.lax.scan`/`jax.tree.map` traverse it transparently.

`Actions`, `SideOutput`, `StepOutput`, `SideTrajectory`, `Trajectory` are
the canonical shapes used by `env.reset`, `env.step`, and `rollout`. Axis
convention everywhere: BATCH → TIME → VEHICLE → FEATURE (outer-to-inner).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

import flax.struct
import jax


class Side(StrEnum):
    """Side discriminator — Python-level enum, never a traced array."""

    GUARD = "guard"
    BANDIT = "bandit"

    def opposite(self) -> Side:
        return Side.BANDIT if self is Side.GUARD else Side.GUARD


@flax.struct.dataclass
class BySide:
    """One entry per side. Itself a JAX pytree."""

    guard: Any
    bandit: Any

    def get(self, side: Side) -> Any:
        # Python-level branch on the StrEnum; safe under jit because `side`
        # is static at trace time.
        return self.guard if side is Side.GUARD else self.bandit

    def map(self, fn: Callable[[Any], Any]) -> BySide:
        return BySide(guard=fn(self.guard), bandit=fn(self.bandit))

    def items(self) -> tuple[tuple[Side, Any], tuple[Side, Any]]:
        return ((Side.GUARD, self.guard), (Side.BANDIT, self.bandit))


@flax.struct.dataclass
class Actions:
    """Per-side stacked actions. `.sides.guard` shape (N_g, action_dim);
    `.sides.bandit` shape (N_b, action_dim)."""

    sides: BySide


@flax.struct.dataclass
class SideOutput:
    """Per-step output for one side. Shape rules per `scope`:

      PER_VEHICLE: obs (N_side, obs_dim); reward (N_side,); done (N_side,) bool
      PER_SIDE   : obs (obs_dim,);        reward ();        done () bool
    """

    obs: jax.Array
    reward: jax.Array
    done: jax.Array


@flax.struct.dataclass
class StepOutput:
    """Result of `env.step`. `outputs.guard` and `outputs.bandit` are
    `SideOutput`s. `episode_done` is the scalar termination signal —
    per-side `done` fields are this scalar broadcast for shape uniformity."""

    state: Any  # EnvState — kept Any to avoid circular import
    outputs: BySide
    episode_done: jax.Array
    info: dict


@flax.struct.dataclass
class SideTrajectory:
    """Time-stacked per-side history. Leading axis (T,) prepended to
    SideOutput-shaped fields plus action / policy_state."""

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    policy_state: Any


@flax.struct.dataclass
class Trajectory:
    """Time-stacked rollout. `env_state` has leading T on every leaf;
    `sides.guard` / `sides.bandit` are `SideTrajectory`s. `episode_done`
    is (T,) latched. `controlled_side` is a Python-level enum (not traced)."""

    env_state: Any  # EnvState
    sides: BySide
    episode_done: jax.Array
    controlled_side: Side = flax.struct.field(pytree_node=False, default=Side.GUARD)
