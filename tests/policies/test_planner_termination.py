"""Search returns must stop at episode termination, including within macro steps."""

import jax
import jax.numpy as jnp
import pytest

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.policies.mcts import MCTSPolicy
from orbitalgym.policies.mppi import MPPIPolicy
from orbitalgym.policies.zero import ZeroControl


def build_case(action_repeat=1):
    cfg = make_lady_bandit_guard(dt=10.0, shaping_gain=0.0, separation_cost=0.0)
    env = OrbitalGymEnv(cfg)
    adapter = POMDPAdapter(env, action_repeat=action_repeat, discount=0.9)
    state, _ = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        guards=state.guards.replace(rtn=jnp.array([[1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])),
        bandits=state.bandits.replace(rtn=jnp.array([[0.0, 6.0, 0.0, 0.0, -0.2, 0.0]])),
    )
    opponent = ZeroControl(n_vehicles=1, command_cls=env.guard_command_cls)
    return env, adapter, state, opponent


@pytest.mark.parametrize("action_repeat", [1, 3])
@pytest.mark.parametrize("independent", [False, True])
def test_mppi_stops_rewards_and_leaf_at_breach(action_repeat, independent):
    env, adapter, state, opponent = build_case(action_repeat)
    policy = MPPIPolicy(
        env_model=adapter,
        side=Side.BANDIT,
        opponent_model=opponent,
        n_samples=2,
        horizon=3,
        terminal_value_fn=lambda s: jnp.asarray(123.0),
        teammate_model=ZeroControl(n_vehicles=1, command_cls=env.bandit_command_cls),
        n_vehicles=1,
        command_cls=env.bandit_command_cls,
    )
    cost = jax.jit(policy._rollout_cost)(
        adapter.pack(state),
        jnp.zeros((3, 3) if independent else (3, 1, 3)),
        jax.random.PRNGKey(1),
        0 if independent else None,
    )
    assert float(cost) == pytest.approx(-1000.0)


@pytest.mark.parametrize("action_repeat", [1, 3])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("scheduled", [False, True])
def test_mcts_stops_discount_and_leaf_at_breach(action_repeat, independent, scheduled):
    env, adapter, state, opponent = build_case(action_repeat)
    from orbitalgym.groundstations.network import ContactSchedule

    schedule = (
        ContactSchedule(
            windows=jnp.array([[0.0, 10000.0]]),
            n_valid=jnp.asarray(1),
            station_ix=jnp.array([0]),
        )
        if scheduled
        else None
    )
    policy = MCTSPolicy(
        env_model=adapter,
        side=Side.BANDIT,
        opponent_model=opponent,
        action_grid=jnp.zeros((1, 3)),
        opponent_action_grid=jnp.zeros((1, 3)),
        num_simulations=4,
        max_depth=3,
        variant="muzero",
        opponent_schedule=schedule,
        teammate_model=ZeroControl(n_vehicles=1, command_cls=env.bandit_command_cls),
        leaf_value_fn=lambda s: jnp.asarray(123.0),
        n_vehicles=1,
        command_cls=env.bandit_command_cls,
    )
    search = (
        (lambda s, k: policy._search_one_vehicle_out(s, k, 0))
        if independent
        else policy._search_joint_out
    )
    out = jax.jit(search)(adapter.pack(state), jax.random.PRNGKey(1))
    tree = out.search_tree
    assert float(tree.children_rewards[0, 0, 0]) == pytest.approx(1000.0)
    assert float(tree.children_discounts[0, 0, 0]) == 0.0
    child = int(tree.children_index[0, 0, 0])
    assert float(tree.raw_values[0, child]) == 0.0
    assert float(tree.qvalues(jnp.array([0]))[0, 0]) == pytest.approx(1000.0)
    # mctx can expand below a terminal node; those descendants stay absorbing.
    expanded = tree.children_index[0, 1:, 0] >= 0
    assert jnp.all(tree.children_rewards[0, 1:, 0][expanded] == 0.0)
    visited = tree.node_visits[0, 1:] > 0
    states = tree.embeddings[0][0]
    assert jnp.all(states[1:][visited] == states[child])


@pytest.mark.parametrize("action_repeat,terminal_substep", [(1, 1), (3, 1), (3, 2), (3, 3)])
def test_adapter_reports_terminal_on_any_macro_substep(action_repeat, terminal_substep):
    env, adapter, state, _ = build_case(action_repeat)
    # Delay the breach to each substep, including the final substep.
    state = state.replace(
        bandits=state.bandits.replace(
            rtn=jnp.array([[0.0, 4.0 + 2.0 * terminal_substep, 0.0, 0.0, -0.2, 0.0]])
        )
    )
    action = jnp.zeros(adapter.guard_action_flat_dim + adapter.bandit_action_flat_dim)
    s_next, reward, done = jax.jit(adapter.step_with_termination, static_argnums=3)(
        adapter.pack(state), action, jax.random.PRNGKey(2), Side.BANDIT
    )
    assert bool(done)
    assert int(adapter.unpack(s_next).step) == terminal_substep
    assert float(reward) == pytest.approx(1000.0 * 0.9 ** (terminal_substep - 1))
    legacy_state, legacy_reward = adapter.step(
        adapter.pack(state), action, jax.random.PRNGKey(2), Side.BANDIT
    )
    assert jnp.allclose(s_next, legacy_state)
    assert float(legacy_reward) == pytest.approx(float(reward))


def test_legacy_adapter_continues_without_termination_metadata():
    from orbitalgym.policies._rollout import step_with_termination

    class LegacyAdapter:
        def step(self, state, action, key, side):
            return state + action, jnp.asarray(2.0)

    state, reward, done = step_with_termination(
        LegacyAdapter(), jnp.array([1.0]), jnp.array([3.0]), jax.random.PRNGKey(0), Side.BANDIT
    )
    assert float(state[0]) == 4.0
    assert float(reward) == 2.0
    assert not bool(done)


@pytest.mark.parametrize("event", ["catch", "horizon"])
def test_mppi_masks_leaf_at_other_terminal_events(event):
    env, adapter, state, opponent = build_case()
    state = state.replace(
        bandits=state.bandits.replace(rtn=jnp.array([[0.0, 200.0, 0.0, 0.0, 0.0, 0.0]]))
    )
    if event == "catch":
        state = state.replace(guards=state.guards.replace(rtn=state.bandits.rtn))
        expected = -1000.0
    else:
        state = state.replace(step=jnp.asarray(env.config.max_steps - 1, dtype=jnp.int32))
        expected = 0.0
    policy = MPPIPolicy(
        env_model=adapter,
        side=Side.BANDIT,
        opponent_model=opponent,
        n_samples=2,
        horizon=3,
        terminal_value_fn=lambda s: jnp.asarray(123.0),
        n_vehicles=1,
        command_cls=env.bandit_command_cls,
    )
    cost = jax.jit(policy._rollout_cost)(
        adapter.pack(state), jnp.zeros((3, 1, 3)), jax.random.PRNGKey(1)
    )
    assert -float(cost) == pytest.approx(expected)
