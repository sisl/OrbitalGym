"""CompositeActionPolicy — sum of base and offset policy actions.

Useful for "scripted base + learned residual" architectures where the
learned head provides corrections on top of a known-reasonable scripted
controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from orbital_game.policies.base import Policy


@dataclass(frozen=True)
class CompositeActionPolicy:
    """Sums actions from `base` and `offset`. Both must produce same-shape actions.

    Only the `dv` field is summed. Non-`dv` Command fields (e.g. `Communicate.active`,
    `Communicate.payload`) propagate from the **base** policy unchanged — the
    offset policy's non-`dv` fields are discarded. This matches the typical
    "scripted base + learned residual" usage where the residual is a Δv
    correction only.
    """

    base: Policy
    offset: Policy
    n_vehicles: int = 0
    command_cls: Any = None

    def __call__(
        self,
        policy_state: Any,
        agent_view: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        if self.command_cls is None:
            raise ValueError(
                "CompositeActionPolicy was called before the env injected "
                "`command_cls`. Use this policy via OrbitalGameEnv / "
                "SingleAgentView, or pass command_cls explicitly."
            )
        k1, k2 = jax.random.split(key)
        base_cmd, _ = self.base(None, agent_view, k1, t)
        offset_cmd, _ = self.offset(None, agent_view, k2, t)
        # Start from base_cmd to preserve non-dv fields; only update dv.
        return base_cmd.replace(dv=base_cmd.dv + offset_cmd.dv), policy_state
