"""Tests for rewards/reference.py — DistanceToReferenceOrbit."""

import jax.numpy as jnp

from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.rewards.base import RewardScope
from orbitalgym.rewards.reference import DistanceToReferenceOrbit


class _Guard:
    rtn = jnp.array([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0]])


class _Env:
    guards = _Guard()


def _make_action_wrapper():
    guard_action = jnp.zeros((2, 3))
    bandit_action = jnp.zeros((1, 3))
    return Actions(sides=BySide(guard=guard_action, bandit=bandit_action))


def test_distance_to_reference_orbit_reward_is_negative_distance():
    """Reference reward: negative sum of guard distances to the reference orbit origin."""
    rw = DistanceToReferenceOrbit()
    action_wrapper = _make_action_wrapper()
    r = rw(
        prev_state=None,
        action=action_wrapper,
        next_state=_Env(),
        side=Side.GUARD,
        params=None,
        t=jnp.array(0.0),
    )
    assert jnp.isclose(r, -(100.0 + 200.0))


def test_distance_to_reference_orbit_scope():
    fn = DistanceToReferenceOrbit()
    assert fn.scope is RewardScope.PER_SIDE


def test_distance_to_reference_orbit_bandit_side_returns_zero():
    fn = DistanceToReferenceOrbit()
    action_wrapper = _make_action_wrapper()
    r = fn(
        prev_state=None,
        action=action_wrapper,
        next_state=_Env(),
        side=Side.BANDIT,
        params=None,
        t=jnp.array(0.0),
    )
    assert float(r) == 0.0
