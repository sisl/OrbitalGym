"""Dynamics protocol shape."""

from __future__ import annotations

from typing import Any, Protocol

import jax


class AppliedControl(Protocol):
    """Output of an Actuator's apply(); consumed by Dynamics."""

    dv: jax.Array  # (n_vehicles, action_dim) — velocity impulse to apply this step


class Dynamics(Protocol):
    def __call__(
        self,
        state: Any,
        applied: AppliedControl,
        params: Any,
        dt: float,
    ) -> Any: ...
