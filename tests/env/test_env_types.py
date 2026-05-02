"""Tests for env/types — pytree registration, BySide accessors, axis-shape conventions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.env.types import (
    Actions,
    BySide,
    Side,
    SideOutput,
    SideTrajectory,
    StepOutput,
    Trajectory,
)


def test_side_opposite():
    assert Side.GUARD.opposite() is Side.BANDIT
    assert Side.BANDIT.opposite() is Side.GUARD


def test_byside_get():
    bs = BySide(guard=1, bandit=2)
    assert bs.get(Side.GUARD) == 1
    assert bs.get(Side.BANDIT) == 2


def test_byside_map():
    bs = BySide(guard=1, bandit=2)
    out = bs.map(lambda x: x * 10)
    assert out.guard == 10
    assert out.bandit == 20


def test_byside_items():
    bs = BySide(guard="a", bandit="b")
    items = bs.items()
    assert items[0] == (Side.GUARD, "a")
    assert items[1] == (Side.BANDIT, "b")


def test_byside_is_pytree():
    bs = BySide(guard=jnp.array([1.0, 2.0]), bandit=jnp.array([3.0, 4.0]))
    leaves, treedef = jax.tree_util.tree_flatten(bs)
    assert len(leaves) == 2
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.guard.shape == (2,)
    assert rebuilt.bandit.shape == (2,)


def test_byside_vmap():
    """vmap over a leading batch axis traverses BySide transparently."""

    def add(bs):
        return BySide(guard=bs.guard + 1, bandit=bs.bandit + 1)

    batched = BySide(
        guard=jnp.zeros((3, 2)),
        bandit=jnp.zeros((3, 2)),
    )
    out = jax.vmap(add)(batched)
    assert out.guard.shape == (3, 2)
    assert (out.guard == 1.0).all()


def test_actions_pytree():
    actions = Actions(
        sides=BySide(
            guard=jnp.zeros((1, 3)),
            bandit=jnp.zeros((2, 3)),
        )
    )
    leaves, treedef = jax.tree_util.tree_flatten(actions)
    assert len(leaves) == 2
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.sides.guard.shape == (1, 3)
    assert rebuilt.sides.bandit.shape == (2, 3)


def test_side_output_pytree():
    so = SideOutput(
        obs=jnp.zeros((1, 6)),
        reward=jnp.zeros((1,)),
        done=jnp.zeros((1,), dtype=bool),
    )
    leaves, treedef = jax.tree_util.tree_flatten(so)
    assert len(leaves) == 3
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.obs.shape == (1, 6)


def test_step_output_pytree():
    so = SideOutput(
        obs=jnp.zeros((1, 6)),
        reward=jnp.zeros((1,)),
        done=jnp.zeros((1,), dtype=bool),
    )
    out = StepOutput(
        state={"t": jnp.array(0.0)},
        outputs=BySide(guard=so, bandit=so),
        episode_done=jnp.array(False),
        info={},
    )
    leaves, _treedef = jax.tree_util.tree_flatten(out)
    # 1 (state.t) + 3*2 (outputs leaves) + 1 (episode_done) = 8 minimum
    assert len(leaves) >= 8


def test_trajectory_controlled_side_is_static():
    """controlled_side is pytree_node=False — not a traced leaf, just metadata."""
    traj = Trajectory(
        env_state={"t": jnp.zeros((10,))},
        sides=BySide(
            guard=SideTrajectory(
                obs=jnp.zeros((10, 1, 6)),
                action=jnp.zeros((10, 1, 3)),
                reward=jnp.zeros((10, 1)),
                done=jnp.zeros((10, 1), dtype=bool),
                policy_state=None,
            ),
            bandit=SideTrajectory(
                obs=jnp.zeros((10, 1, 6)),
                action=jnp.zeros((10, 1, 3)),
                reward=jnp.zeros((10, 1)),
                done=jnp.zeros((10, 1), dtype=bool),
                policy_state=None,
            ),
        ),
        episode_done=jnp.zeros((10,), dtype=bool),
        controlled_side=Side.GUARD,
    )
    leaves, _treedef = jax.tree_util.tree_flatten(traj)
    assert all(not isinstance(leaf, Side) for leaf in leaves)
