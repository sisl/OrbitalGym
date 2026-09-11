"""Root candidate limits control actual search coverage in both MCTS paths."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.policies.mcts import MCTSPolicy
from orbitalgym.policies.uniform_random import UniformRandomDiscretePolicy


def _search_setup(coordination):
    cfg = make_lady_bandit_guard(seed=0, n_guards=2, n_bandits=1)
    env = OrbitalGymEnv(cfg)
    adapter = POMDPAdapter(env)
    # Joint searches have 9**2=81 choices; independent searches have 27.
    n_actions = 9 if coordination == "joint" else 27
    angles = jnp.arange(n_actions) * (2 * jnp.pi / n_actions)
    grid = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    opponent = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=1, command_cls=env.bandit_command_cls
    )
    teammate = UniformRandomDiscretePolicy(
        action_grid=grid, n_vehicles=2, command_cls=env.guard_command_cls
    )
    policy = MCTSPolicy(
        env_model=adapter,
        side=Side.GUARD,
        action_grid=grid,
        opponent_model=opponent,
        opponent_action_grid=grid,
        num_simulations=64,
        max_depth=1,
        n_vehicles=2,
        command_cls=env.guard_command_cls,
        coordination=coordination,
        teammate_model=teammate,
    )
    state, _ = env.reset(jax.random.key(0))
    return policy, adapter.pack(state)


def _root_visits(policy, state):
    key = jax.random.key(1)
    if policy.coordination == "joint":
        result = policy._search_joint_out(state, key)
    else:
        result = policy._search_one_vehicle_out(state, key, self_index=1)
    return np.asarray(result.search_tree.children_visits[0, 0])


@pytest.mark.parametrize("coordination", ["joint", "independent"])
def test_candidate_budget_changes_root_coverage_at_fixed_simulation_count(coordination):
    """Omitting either forwarding argument leaves coverage stuck at 16."""
    policy, state = _search_setup(coordination)
    small = dataclasses.replace(policy, max_num_considered_actions=4)
    large = dataclasses.replace(policy, max_num_considered_actions=24)

    small_visits = _root_visits(small, state)
    large_visits = _root_visits(large, state)

    assert small_visits.sum() == large_visits.sum() == 64
    assert 0 < np.count_nonzero(small_visits) <= 4
    assert 16 < np.count_nonzero(large_visits) <= 24


@pytest.mark.parametrize("coordination", ["joint", "independent"])
def test_default_candidate_budget_preserves_sixteen_root_actions(coordination):
    policy, state = _search_setup(coordination)
    visits = _root_visits(policy, state)
    assert visits.sum() == 64
    assert np.count_nonzero(visits) == 16


@pytest.mark.parametrize("limit", [0, -1])
def test_candidate_budget_rejects_nonpositive_limits(limit):
    policy, _ = _search_setup("joint")
    with pytest.raises(ValueError, match="max_num_considered_actions.*positive"):
        dataclasses.replace(policy, max_num_considered_actions=limit)
