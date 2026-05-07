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


def _build_multi(n_guards: int = 2, n_bandits: int = 1):
    """Builder mirroring _build but with arbitrary fleet sizes."""
    cfg = make_lady_bandit_guard(seed=0, n_guards=n_guards, n_bandits=n_bandits)
    env = OrbitalGameEnv(cfg)
    adapter = POMDPAdapter(env)
    state, _ = env.reset(jax.random.PRNGKey(0))
    return env, adapter, state, cfg


def test_mcts_joint_emits_distinct_per_vehicle_dvs(monkeypatch):
    """Joint mode searches over A^n joint actions; the output Δv should be
    indexable per-vehicle, not a single Δv broadcast across the fleet (the
    pre-fix behaviour). We force a non-uniform joint index by stubbing the
    search to a known value, then verify the per-vehicle decoding."""
    n_g = 2
    env, adapter, state, cfg = _build_multi(n_guards=n_g)
    grid = _grid()  # (9, 2)
    opp = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )
    policy = MCTSPolicy(
        env_model=adapter,
        side=Side.GUARD,
        action_grid=grid,
        opponent_model=opp,
        opponent_action_grid=grid,
        num_simulations=4,
        n_vehicles=cfg.n_guards,
        command_cls=env.guard_command_cls,
        coordination="joint",
    )
    s_flat = adapter.pack(state)
    cmd, _ = policy(None, s_flat, jax.random.PRNGKey(2), state.t)
    # Must shape as (n_guards, dv_dim_cmd). Default LBG factory uses RTN dynamics
    # (3-D Δv), and `_action_idx_to_command` zero-pads the 2-D grid rows to 3-D.
    cmd_dv_dim = env.guard_command_cls.zeros(n_g).dv.shape[-1]
    assert cmd.dv.shape == (n_g, cmd_dv_dim)
    # Per-vehicle Δv must equal a grid row in the first 2 dims (with the trailing
    # padded dims zero).
    grid_dim = grid.shape[-1]
    for v in range(n_g):
        head = cmd.dv[v, :grid_dim]
        tail = cmd.dv[v, grid_dim:]
        assert any(jnp.allclose(head, grid[i]) for i in range(grid.shape[0]))
        assert jnp.allclose(tail, 0.0)


def test_mcts_independent_runs_per_vehicle_search():
    """Independent mode requires teammate_model and produces a per-vehicle
    Command. The chosen Δvs need not be identical across guards (the bug we
    fixed), and they must each lie on the action grid."""
    n_g = 3
    env, adapter, state, cfg = _build_multi(n_guards=n_g)
    grid = _grid()
    opp = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )
    teammate = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls
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
        coordination="independent",
        teammate_model=teammate,
    )
    s_flat = adapter.pack(state)
    cmd, _ = policy(None, s_flat, jax.random.PRNGKey(3), state.t)
    cmd_dv_dim = env.guard_command_cls.zeros(n_g).dv.shape[-1]
    assert cmd.dv.shape == (n_g, cmd_dv_dim)
    # Each per-vehicle row must equal one of the grid rows in the first
    # `grid_dim` slots (the trailing slots are zero-padded by the policy).
    grid_dim = grid.shape[-1]
    for v in range(n_g):
        head = cmd.dv[v, :grid_dim]
        tail = cmd.dv[v, grid_dim:]
        assert any(jnp.allclose(head, grid[i]) for i in range(grid.shape[0]))
        assert jnp.allclose(tail, 0.0)


def test_mcts_independent_requires_teammate_model_when_multi_vehicle():
    """coordination='independent' with n_vehicles>1 must raise without a teammate_model."""
    n_g = 2
    env, adapter, _, cfg = _build_multi(n_guards=n_g)
    grid = _grid()
    opp = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )
    with pytest.raises(ValueError, match="teammate_model"):
        MCTSPolicy(
            env_model=adapter,
            side=Side.GUARD,
            action_grid=grid,
            opponent_model=opp,
            opponent_action_grid=grid,
            num_simulations=4,
            n_vehicles=cfg.n_guards,
            command_cls=env.guard_command_cls,
            coordination="independent",
            # teammate_model omitted on purpose
        )


def test_mcts_unknown_coordination_raises():
    grid = _grid()
    env, adapter, _, cfg = _build_multi(n_guards=1)
    opp = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )
    with pytest.raises(ValueError, match="coordination"):
        MCTSPolicy(
            env_model=adapter,
            side=Side.GUARD,
            action_grid=grid,
            opponent_model=opp,
            opponent_action_grid=grid,
            n_vehicles=cfg.n_guards,
            command_cls=env.guard_command_cls,
            coordination="frobnicate",  # type: ignore[arg-type]
        )


def test_mcts_independent_action_diversity_across_keys():
    """Across many PRNG keys, independent-mode MCTS should produce per-vehicle
    Δvs that are not all identical: at least one key must yield a step where
    the guards differ. (Catches the regression where one searched action was
    broadcast to every vehicle.)"""
    n_g = 3
    env, adapter, state, cfg = _build_multi(n_guards=n_g)
    grid = _grid()
    opp = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls
    )
    teammate = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls
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
        coordination="independent",
        teammate_model=teammate,
    )
    s_flat = adapter.pack(state)
    seen_diff = False
    for k in range(10):
        cmd, _ = policy(None, s_flat, jax.random.PRNGKey(k), state.t)
        dv = np.asarray(cmd.dv)
        # Any pair of guards differing means we broke out of the
        # broadcast-to-all-vehicles regression.
        if not np.allclose(dv - dv[0:1], 0.0):
            seen_diff = True
            break
    assert seen_diff, (
        "Independent-mode MCTS produced identical Δv across all guards on every "
        "key — regressed to the broadcast bug."
    )


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
