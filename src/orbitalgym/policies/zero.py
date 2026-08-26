"""ZeroControl — emits the identity Command pytree for the policy's side.

Default policy for both sides; reference for bootstrap validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from orbitalgym.registry import PolicyKey, register


@register(PolicyKey.ZERO_CONTROL)
@dataclass(frozen=True)
class ZeroControl:
    """Always emits the identity Command (zeros for every component).

    Env-populated init knobs (filled by OrbitalGymEnv via dataclasses.replace):
        command_cls: the per-side Command class (built by build_command_class)
        n_vehicles: number of vehicles on this side

    Replaces the previous (n_vehicles, action_dim) raw-array contract.
    """

    command_cls: Any = None
    n_vehicles: int = 0

    def __call__(
        self,
        policy_state: Any,
        agent_view: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        del agent_view, key, t
        if self.command_cls is None:
            raise ValueError(
                "ZeroControl was called before the env injected `command_cls`. "
                "Use this policy via OrbitalGymEnv / SingleAgentView, or pass "
                "command_cls explicitly: "
                "ZeroControl(command_cls=cmd_cls, n_vehicles=N)."
            )
        return self.command_cls.zeros(self.n_vehicles), policy_state
