"""PointingPolicy fills target_dir from the belief mean."""

import flax
import jax
import jax.numpy as jnp

from orbitalgym.policies.pointing import PointingPolicy, PointingTarget


@flax.struct.dataclass
class _Belief:
    mean: jax.Array


@flax.struct.dataclass
class _Cmd:
    dv: jax.Array
    target_dir: jax.Array


def _inner(policy_state, agent_view, key, t):
    n = agent_view.mean.shape[0]
    return _Cmd(dv=jnp.zeros((n, 3)), target_dir=jnp.zeros((n, 3))), policy_state


def _belief():
    # Two guards at (0, 100, 0) and (0, -100, 0); one bandit at (500, 0, 0).
    rows = jnp.array(
        [
            [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -100.0, 0.0, 0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return _Belief(mean=jnp.broadcast_to(rows[None], (2, 3, 6)))


def test_lady_target_points_to_origin():
    cmd, _ = PointingPolicy(inner=_inner, target=PointingTarget.LADY)(None, _belief(), None, 0.0)
    assert jnp.allclose(cmd.target_dir[0], jnp.array([0.0, -1.0, 0.0]), atol=1e-6)


def test_opponent_target_points_to_nearest_bandit():
    cmd, _ = PointingPolicy(inner=_inner, target=PointingTarget.OPPONENT_BELIEF)(
        None, _belief(), None, 0.0
    )
    expected = jnp.array([500.0, -100.0, 0.0])
    assert jnp.allclose(cmd.target_dir[0], expected / jnp.linalg.norm(expected), atol=1e-6)


def test_teammate_target_points_to_nearest_teammate():
    cmd, _ = PointingPolicy(inner=_inner, target=PointingTarget.TEAMMATE)(
        None, _belief(), None, 0.0
    )
    assert jnp.allclose(cmd.target_dir[0], jnp.array([0.0, -1.0, 0.0]), atol=1e-6)
    assert jnp.allclose(cmd.target_dir[1], jnp.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_hold_target_is_zero():
    cmd, _ = PointingPolicy(inner=_inner, target=PointingTarget.HOLD)(None, _belief(), None, 0.0)
    assert jnp.allclose(cmd.target_dir, 0.0)
