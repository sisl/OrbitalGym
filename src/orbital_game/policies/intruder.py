"""Reference intruder policy."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from orbital_game.registry import IntruderPolicyKey, register


@register(IntruderPolicyKey.ZERO_CONTROL)
@dataclass(frozen=True)
class ZeroControlIntruder:
    """Always outputs a zero command. Reference for bootstrap validation."""

    n_intruders: int
    action_dim: int = 3

    def __call__(self, obs, key, t):
        del obs, key, t
        return jnp.zeros((self.n_intruders, self.action_dim))
