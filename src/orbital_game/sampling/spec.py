"""ICSpec — top-level initial-condition specification.

Composes per-side samplers + validators + global max_attempts. Validators
are tuples (immutable, hashable for tracing). Default max_attempts=100 is
generous; downstream code can monitor traj.env_state.ic_valid to detect
real-world cap exhaustion rates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax

from orbital_game.registry import StateComponentKey

if TYPE_CHECKING:
    from orbital_game.config import ScenarioConfig


@runtime_checkable
class SideSampler(Protocol):
    """Sample initial state for one side (defenders or intruders).

    Concrete implementations are frozen dataclasses with __call__. The env
    passes (n_vehicles, components, class_name) as keyword args so the same
    sampler instance can serve either side without baking side identity in.
    """

    def __call__(
        self,
        config: ScenarioConfig,
        key: jax.Array,
        *,
        n_vehicles: int,
        components: tuple[StateComponentKey, ...],
        class_name: str,
    ) -> Any: ...


@runtime_checkable
class Validator(Protocol):
    """Post-sample combined-state check. Returns scalar bool jax.Array."""

    def __call__(
        self,
        config: ScenarioConfig,
        defenders: Any,
        intruders: Any,
    ) -> jax.Array: ...


@dataclass(frozen=True)
class ICSpec:
    """Composable initial-condition specification.

    Used as `ScenarioConfig.ic_sampler`. The env reads this once at reset,
    runs validators in a `jax.lax.while_loop` capped at `max_attempts`, and
    sets `EnvState.ic_valid=False` if the cap is hit.
    """

    defender_sampler: Callable
    intruder_sampler: Callable
    validators: tuple[Callable, ...] = ()
    max_attempts: int = 100
