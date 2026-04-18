"""Actuator protocol + AppliedControl dataclass."""

from __future__ import annotations

import flax.struct
import jax


@flax.struct.dataclass
class AppliedControl:
    dv: jax.Array  # (n, action_dim) — velocity impulse applied at start of the step
