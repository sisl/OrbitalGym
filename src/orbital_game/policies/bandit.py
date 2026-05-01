"""Reference bandit policy."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from orbital_game.registry import BanditPolicyKey, register


@register(BanditPolicyKey.ZERO_CONTROL)
@dataclass(frozen=True)
class ZeroControlBandit:
    """Always outputs a zero command. Reference for bootstrap validation."""

    n_bandits: int
    action_dim: int = 3

    def __call__(self, obs, key, t):
        del obs, key, t
        return jnp.zeros((self.n_bandits, self.action_dim))
