"""Tests for rewards/reference.py — DistanceToReferenceOrbit."""

import jax.numpy as jnp

from orbital_game.rewards.reference import DistanceToReferenceOrbit


def test_distance_to_reference_orbit_reward_is_negative_distance():
    """Reference reward: negative sum of guard distances to the reference orbit origin."""

    class _Guard:
        rtn = jnp.array([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0]])

    class _Env:
        guards = _Guard()

    rw = DistanceToReferenceOrbit()
    r = rw(prev_state=None, action=None, next_state=_Env(), params=None, t=jnp.array(0.0))
    assert jnp.isclose(r, -(100.0 + 200.0))
