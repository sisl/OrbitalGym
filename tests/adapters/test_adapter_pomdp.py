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
