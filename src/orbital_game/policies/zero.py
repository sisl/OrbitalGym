"""ZeroControl — always emits a zero command. The default scripted policy
for both sides; reference for bootstrap validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbital_game.registry import PolicyKey, register


@register(PolicyKey.ZERO_CONTROL)
@dataclass(frozen=True)
class ZeroControl:
    """Always emits a zero command. Reference for bootstrap validation
    and the default scripted policy for both sides."""

    # Env-populated dims (filled by OrbitalGameEnv.__init__ via dataclasses.replace):
    n_vehicles: int = 0
    action_dim: int = 0

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, Any]:
        del obs, key, t
        return jnp.zeros((self.n_vehicles, self.action_dim)), policy_state
