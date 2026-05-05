"""LBG reward with per-broadcast communication cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from orbital_game.env.types import Actions, Side
from orbital_game.registry import RewardFnKey, register
from orbital_game.rewards.base import RewardScope
from orbital_game.rewards.reference import DistanceToReferenceOrbit


@register(RewardFnKey.LBG_WITH_COMMS)
@dataclass(frozen=True)
class LbgWithCommsReward:
    """Distance-to-reference + per-broadcast comm cost."""

    base_reward: DistanceToReferenceOrbit = field(default_factory=DistanceToReferenceOrbit)
    comm_cost: float = 5.0
    scope: RewardScope = RewardScope.PER_SIDE

    def __call__(
        self,
        prev_state: Any,
        action: Actions,
        next_state: Any,
        side: Side,
        params: Any,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        base = self.base_reward(prev_state, action, next_state, side, params, t)
        if side is Side.GUARD:
            # Bandits don't communicate in LBG, so the bandit reward path
            # short-circuits without touching `active`. The guard side, by
            # contract, must wire the Communicate component (env-construction
            # preflight in OrbitalGameEnv enforces this when LbgWithCommsReward
            # is configured); a missing `active` field raises clearly here.
            n_active = jnp.sum(action.sides.guard.active.astype(jnp.float32))
            return base - self.comm_cost * n_active
        return base
