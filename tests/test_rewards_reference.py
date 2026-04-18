"""Tests for rewards/reference.py — DistanceToHVA."""

import jax.numpy as jnp

from orbital_game.rewards.reference import DistanceToHVA


def test_distance_to_hva_reward_is_negative_distance():
    """Reference reward: sum of negative Euclidean distances from each defender to HVA."""

    class _Def:
        rtn = jnp.array([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0]])

    class _Env:
        defenders = _Def()

    rw = DistanceToHVA()
    r = rw(prev_state=None, action=None, next_state=_Env(), params=None, t=jnp.array(0.0))
    assert jnp.isclose(r, -(100.0 + 200.0))
