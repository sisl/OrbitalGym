"""Heuristic leaf values for MCTS."""

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.policies.leaf_values import bandit_leaf_value, guard_leaf_value


def _state_with(guard_r, bandit_r):
    env = OrbitalGymEnv(make_lady_bandit_guard())
    adapter = POMDPAdapter(env)
    state, _ = env.reset(jax.random.PRNGKey(0))
    guards = state.guards.replace(rtn=jnp.array([[guard_r, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    bandits = state.bandits.replace(rtn=jnp.array([[bandit_r, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    return adapter, adapter.pack(state.replace(guards=guards, bandits=bandits))


def test_guard_value_prefers_close_guard_and_far_bandit():
    adapter, near = _state_with(guard_r=950.0, bandit_r=1000.0)
    _, far = _state_with(guard_r=100.0, bandit_r=1000.0)
    value = guard_leaf_value(adapter, scale_m=100.0)
    assert float(value(near)) > float(value(far))
    assert jnp.allclose(value(near), (1000.0 - 50.0) / 100.0)


def test_bandit_value_prefers_close_to_lady():
    adapter, close = _state_with(guard_r=0.0, bandit_r=100.0)
    _, distant = _state_with(guard_r=0.0, bandit_r=1000.0)
    value = bandit_leaf_value(adapter, scale_m=100.0)
    assert float(value(close)) > float(value(distant))
    assert jnp.allclose(value(close), -1.0)
