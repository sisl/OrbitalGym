"""Tests for POMDPAdapter — protocol shape conformance + JAX consistency."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from examples.reference_scenario import build_config
from orbitalgym.adapters.pomdp import POMDPAdapter
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Side


def _make_adapter():
    cfg = build_config()
    env = OrbitalGymEnv(cfg)
    return POMDPAdapter(env), env, cfg


def test_pomdp_adapter_states_dim_matches_layout():
    adapter, env, _cfg = _make_adapter()
    # +2 for the (t, step) tail packed onto the flat vector.
    assert adapter.states_dim == env.layout.flat_dim + 2


def test_pomdp_adapter_discount_is_one():
    adapter, _env, _cfg = _make_adapter()
    assert adapter.discount() == 1.0


def test_pomdp_adapter_initialstate_returns_flat_vector():
    adapter, _env, _cfg = _make_adapter()
    s = adapter.initialstate(jax.random.PRNGKey(0))
    assert s.shape == (adapter.states_dim,)


def test_pomdp_adapter_transition_advances_state():
    adapter, _env, cfg = _make_adapter()
    s0 = adapter.initialstate(jax.random.PRNGKey(0))
    n_g, n_b = cfg.n_guards, cfg.n_bandits
    d = adapter.action_dim_per_side
    a = jnp.zeros(n_g * d + n_b * d)
    s1 = adapter.transition(s0, a, jax.random.PRNGKey(1))
    assert s1.shape == s0.shape
    assert not bool(jnp.allclose(s0, s1))


def test_pomdp_adapter_per_side_reward():
    adapter, _env, cfg = _make_adapter()
    s0 = adapter.initialstate(jax.random.PRNGKey(0))
    n_g, n_b = cfg.n_guards, cfg.n_bandits
    d = adapter.action_dim_per_side
    a = jnp.zeros(n_g * d + n_b * d)
    s1 = adapter.transition(s0, a, jax.random.PRNGKey(1))
    r_g = adapter.reward(s0, a, s1, Side.GUARD)
    r_b = adapter.reward(s0, a, s1, Side.BANDIT)
    assert r_g.shape == ()
    assert r_b.shape == ()


def test_pomdp_adapter_observation_per_side():
    adapter, _env, cfg = _make_adapter()
    s0 = adapter.initialstate(jax.random.PRNGKey(0))
    n_g, n_b = cfg.n_guards, cfg.n_bandits
    d = adapter.action_dim_per_side
    a = jnp.zeros(n_g * d + n_b * d)
    s1 = adapter.transition(s0, a, jax.random.PRNGKey(1))
    o_g = adapter.observation(s0, a, s1, Side.GUARD)
    o_b = adapter.observation(s0, a, s1, Side.BANDIT)
    assert o_g.ndim == 1
    assert o_b.ndim == 1


def test_pomdp_adapter_transition_is_pure_under_vmap():
    """Adapter is pure: transition composes with jax.vmap without leaking
    tracers (regression test for the old stateful _last_state design)."""
    adapter, _env, cfg = _make_adapter()
    s0 = adapter.initialstate(jax.random.PRNGKey(0))
    n_g, n_b = cfg.n_guards, cfg.n_bandits
    d = adapter.action_dim_per_side
    action_dim = (n_g + n_b) * d

    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    actions = jax.random.normal(jax.random.PRNGKey(2), (4, action_dim)) * 0.05

    def step(k, a):
        return adapter.transition(s0, a, k)

    s_next_batch = jax.vmap(step)(keys, actions)
    assert s_next_batch.shape == (4, adapter.states_dim)
    # And a regular post-vmap call must still work — this would raise
    # UnexpectedTracerError under the old stateful design.
    s_next = adapter.transition(s0, actions[0], keys[0])
    assert s_next.shape == (adapter.states_dim,)


def _macro_env(action_repeat: int, discount: float = 0.99):
    cfg = build_config()
    env = OrbitalGymEnv(cfg)
    return POMDPAdapter(env, action_repeat=action_repeat, discount=discount), env, cfg


def _action(adapter, cfg, value=0.01):
    d = adapter.action_dim_per_side
    return jnp.full((cfg.n_guards + cfg.n_bandits) * d, value)


def test_pomdp_adapter_macro_dt_and_repeat_exposed():
    adapter, _env, cfg = _macro_env(3)
    assert adapter.action_repeat == 3
    assert adapter.macro_dt == cfg.dt * 3
    assert adapter.discount() == 0.99**3


def test_pomdp_adapter_macro_transition_matches_three_env_steps():
    adapter, env, cfg = _macro_env(3)
    s0 = adapter.initialstate(jax.random.PRNGKey(0))
    a = _action(adapter, cfg)
    key = jax.random.PRNGKey(1)
    actions = adapter._make_actions(a)

    state = adapter.unpack(s0)
    for k in jax.random.split(key, 3):
        state = env.step(k, state, actions).state

    assert jnp.allclose(adapter.transition(s0, a, key), adapter.pack(state))


def test_pomdp_adapter_macro_reward_is_discounted_substep_sum():
    adapter, env, cfg = _macro_env(3)
    s0 = adapter.initialstate(jax.random.PRNGKey(0))
    a = _action(adapter, cfg)
    key = jax.random.PRNGKey(1)
    actions = adapter._make_actions(a)

    state = adapter.unpack(s0)
    expected = 0.0
    for i, k in enumerate(jax.random.split(key, 3)):
        out = env.step(k, state, actions)
        r = env.reward_fn(state, actions, out.state, Side.GUARD, env.config, state.t)
        expected += 0.99**i * float(r)
        state = out.state

    s_next, reward = adapter.step(s0, a, key, Side.GUARD)
    assert jnp.allclose(reward, expected, atol=1e-5)
    assert jnp.allclose(adapter.reward(s0, a, s_next, Side.GUARD), reward)


def test_pomdp_adapter_macro_freezes_state_after_terminal_substep():
    adapter, env, cfg = _macro_env(3)
    a = _action(adapter, cfg)
    key = jax.random.PRNGKey(1)
    actions = adapter._make_actions(a)

    state0, _ = env.reset(jax.random.PRNGKey(0))
    # Two substeps before the horizon: substep 2 terminates, substep 3 is frozen.
    state0 = state0.replace(step=jnp.asarray(cfg.max_steps - 2, dtype=jnp.int32))
    s0 = adapter.pack(state0)

    state = state0
    expected = 0.0
    for i, k in enumerate(jax.random.split(key, 3)[:2]):
        out = env.step(k, state, actions)
        r = env.reward_fn(state, actions, out.state, Side.GUARD, env.config, state.t)
        expected += 0.99**i * float(r)
        state = out.state
    assert bool(out.episode_done)

    s_next, reward = adapter.step(s0, a, key, Side.GUARD)
    assert jnp.allclose(s_next, adapter.pack(state))
    assert jnp.allclose(reward, expected, atol=1e-5)


def test_pomdp_adapter_action_repeat_one_matches_single_env_step():
    adapter, env, cfg = _macro_env(1, discount=1.0)
    s0 = adapter.initialstate(jax.random.PRNGKey(0))
    a = _action(adapter, cfg)
    key = jax.random.PRNGKey(1)
    actions = adapter._make_actions(a)

    out = env.step(key, adapter.unpack(s0), actions)
    s1 = adapter.transition(s0, a, key)
    assert jnp.array_equal(s1, adapter.pack(out.state))
    r = adapter.reward(s0, a, s1, Side.GUARD)
    assert jnp.array_equal(
        r,
        env.reward_fn(
            adapter.unpack(s0), actions, out.state, Side.GUARD, env.config, adapter.unpack(s0).t
        ),
    )
    s_step, r_step = adapter.step(s0, a, key, Side.GUARD)
    assert jnp.array_equal(s_step, s1)
    assert jnp.array_equal(r_step, r)
    # Pinned single-step numbers for the reference scenario at these seeds.
    assert jnp.allclose(r, -0.0012093481163598)
    assert jnp.allclose(s1[0], -998.0730644985427)
