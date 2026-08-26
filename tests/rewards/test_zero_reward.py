"""Tests for ZeroReward — no-op reward for both sides."""

from __future__ import annotations

import jax.numpy as jnp

from orbitalgym.env.types import Side
from orbitalgym.rewards.base import RewardScope
from orbitalgym.rewards.reference import ZeroReward


def test_zero_reward_returns_zero_for_guard():
    fn = ZeroReward()
    out = fn(
        prev_state=None,
        action=None,
        next_state=None,
        side=Side.GUARD,
        params=None,
        t=jnp.asarray(0.0),
    )
    assert float(out) == 0.0


def test_zero_reward_returns_zero_for_bandit():
    fn = ZeroReward()
    out = fn(
        prev_state=None,
        action=None,
        next_state=None,
        side=Side.BANDIT,
        params=None,
        t=jnp.asarray(0.0),
    )
    assert float(out) == 0.0


def test_zero_reward_scope_is_per_side():
    assert ZeroReward().scope is RewardScope.PER_SIDE
