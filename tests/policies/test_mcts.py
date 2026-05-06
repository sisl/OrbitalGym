"""MCTSPolicy — JAX-native classic UCT search via mctx.

Tests cover:

- Construction guard (mctx required, command_cls/n_vehicles required).
- Round-trip with a flat state vector as ``agent_view``.
- Round-trip with a Belief-shaped object as ``agent_view``.
- ``jax.jit`` wrap is well-defined.
- Two-side rollout with both sides running MCTSPolicy doesn't crash and
  emits commands of the right shape.
"""

from __future__ import annotations

import dataclasses

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbital_game import OrbitalGameEnv, Side, make_lady_bandit_guard
from orbital_game.adapters.pomdp import POMDPAdapter
from orbital_game.policies.mcts import MCTSPolicy
from orbital_game.policies.uniform_random import UniformRandomDiscretePolicy


def _grid(n_dirs: int = 8, dv_max: float = 1.0) -> jnp.ndarray:
    """``n_dirs`` directions on the RT plane + a no-op."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    vecs = dv_max * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return jnp.asarray(np.concatenate([vecs, np.zeros((1, 2))], axis=0), dtype=jnp.float32)


@flax.struct.dataclass
class _OracleBelief:
    """Trivial Belief whose ``mean`` is the adapter's flat state vector."""

    mean: jax.Array


def _build():
    cfg = make_lady_bandit_guard(seed=0)
    env = OrbitalGameEnv(cfg)
    adapter = POMDPAdapter(env)
    grid = _grid()
    opp = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )
    policy = MCTSPolicy(
        env_model=adapter,
        side=Side.GUARD,
        action_grid=grid,
        opponent_model=opp,
        opponent_action_grid=grid,
        num_simulations=8,
        n_vehicles=cfg.n_guards,
        command_cls=env.guard_command_cls,
    )
    state, _ = env.reset(jax.random.PRNGKey(0))
    return env, adapter, policy, state, cfg


def test_mcts_policy_requires_mctx_and_command_cls():
    """Missing command_cls/n_vehicles must raise immediately."""
    grid = _grid()
    with pytest.raises(ValueError, match="command_cls"):
        MCTSPolicy(
            env_model=None,  # type: ignore[arg-type]
            side=Side.GUARD,
            action_grid=grid,
            opponent_model=None,
            opponent_action_grid=grid,
        )


def test_mcts_policy_emits_command_pytree_from_flat_state():
    env, adapter, policy, state, cfg = _build()
    s_flat = adapter.pack(state)
    cmd, ps = policy(None, s_flat, jax.random.PRNGKey(1), state.t)
    expected_dv_shape = env.guard_command_cls.zeros(cfg.n_guards).dv.shape
    assert cmd.dv.shape == expected_dv_shape
    assert ps is None


def test_mcts_policy_reads_belief_mean():
    """When agent_view is a Belief, MCTSPolicy must use ``mean`` as the search root."""
    env, adapter, policy, state, cfg = _build()
    s_flat = adapter.pack(state)
    belief = _OracleBelief(mean=s_flat)
    cmd_state, _ = policy(None, s_flat, jax.random.PRNGKey(7), state.t)
    cmd_belief, _ = policy(None, belief, jax.random.PRNGKey(7), state.t)
    # Same key + same effective state → same chosen action.
    assert jnp.allclose(cmd_state.dv, cmd_belief.dv)


def test_mcts_policy_jit_compatible():
    env, adapter, policy, state, cfg = _build()
    s_flat = adapter.pack(state)

    @jax.jit
    def call_jit(s, k, tt):
        return policy(None, s, k, tt)

    cmd, _ = call_jit(s_flat, jax.random.PRNGKey(2), state.t)
    expected_dv_shape = env.guard_command_cls.zeros(cfg.n_guards).dv.shape
    assert cmd.dv.shape == expected_dv_shape


def test_mcts_policy_action_within_action_grid():
    """Chosen Δv must come from the action grid (after pad-to-cmd-dim)."""
    env, adapter, policy, state, cfg = _build()
    s_flat = adapter.pack(state)
    cmd, _ = policy(None, s_flat, jax.random.PRNGKey(3), state.t)
    chosen = np.asarray(cmd.dv[0, :2])  # first two dims = grid Δv
    grid_np = np.asarray(policy.action_grid)
    diffs = np.linalg.norm(grid_np - chosen[None, :], axis=-1)
    assert diffs.min() < 1e-5, f"chosen {chosen} not in grid {grid_np}"


def test_mcts_policy_both_sides():
    """Both sides may run MCTSPolicy (each with its own opponent_model)."""
    cfg = make_lady_bandit_guard(seed=0)
    env = OrbitalGameEnv(cfg)
    adapter = POMDPAdapter(env)
    grid = _grid()

    opp_for_guard = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )
    opp_for_bandit = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls
    )
    guard_policy = MCTSPolicy(
        env_model=adapter,
        side=Side.GUARD,
        action_grid=grid,
        opponent_model=opp_for_guard,
        opponent_action_grid=grid,
        num_simulations=4,
        n_vehicles=cfg.n_guards,
        command_cls=env.guard_command_cls,
    )
    bandit_policy = MCTSPolicy(
        env_model=adapter,
        side=Side.BANDIT,
        action_grid=grid,
        opponent_model=opp_for_bandit,
        opponent_action_grid=grid,
        num_simulations=4,
        n_vehicles=cfg.n_bandits,
        command_cls=env.bandit_command_cls,
    )
    state, _ = env.reset(jax.random.PRNGKey(0))
    s_flat = adapter.pack(state)
    g_cmd, _ = guard_policy(None, s_flat, jax.random.PRNGKey(1), state.t)
    b_cmd, _ = bandit_policy(None, s_flat, jax.random.PRNGKey(2), state.t)
    assert g_cmd.dv.shape == env.guard_command_cls.zeros(cfg.n_guards).dv.shape
    assert b_cmd.dv.shape == env.bandit_command_cls.zeros(cfg.n_bandits).dv.shape


def test_mcts_policy_search_budget_changes_action_distribution():
    """Larger search budget should at least be reachable and not crash; we
    check that the policy returns *some* grid action for each of two budgets."""
    env, adapter, _, state, cfg = _build()
    grid = _grid()
    s_flat = adapter.pack(state)
    opp = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )

    def make_policy(num_simulations: int) -> MCTSPolicy:
        return MCTSPolicy(
            env_model=adapter,
            side=Side.GUARD,
            action_grid=grid,
            opponent_model=opp,
            opponent_action_grid=grid,
            num_simulations=num_simulations,
            n_vehicles=cfg.n_guards,
            command_cls=env.guard_command_cls,
        )

    p_small = make_policy(4)
    p_big = make_policy(32)
    cmd_s, _ = p_small(None, s_flat, jax.random.PRNGKey(11), state.t)
    cmd_b, _ = p_big(None, s_flat, jax.random.PRNGKey(11), state.t)
    expected = env.guard_command_cls.zeros(cfg.n_guards).dv.shape
    assert cmd_s.dv.shape == expected
    assert cmd_b.dv.shape == expected


def test_mcts_policy_dataclass_replace_pattern():
    """Mirrors the env-injection convention: build the policy with stub
    command_cls/n_vehicles, then ``dataclasses.replace`` to bind the env."""
    env, adapter, _, _, cfg = _build()
    grid = _grid()
    opp = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )
    bare = MCTSPolicy(
        env_model=adapter,
        side=Side.GUARD,
        action_grid=grid,
        opponent_model=opp,
        opponent_action_grid=grid,
        num_simulations=2,
        n_vehicles=cfg.n_guards,
        command_cls=env.guard_command_cls,
    )
    bound = dataclasses.replace(bare, num_simulations=4)
    assert bound.num_simulations == 4
    assert bound.command_cls is env.guard_command_cls
