"""Tests for POMDPAdapter — protocol shape conformance + JAX consistency."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from examples.reference_scenario import build_config
from orbital_game.adapters.pomdp import POMDPAdapter
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Side


def _make_adapter():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    return POMDPAdapter(env), env, cfg


def test_pomdp_adapter_states_dim_matches_layout():
    adapter, env, _cfg = _make_adapter()
    assert adapter.states_dim == env.layout.flat_dim


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


def test_pomdp_adapter_transition_before_init_raises():
    adapter, _env, cfg = _make_adapter()
    n_g, n_b = cfg.n_guards, cfg.n_bandits
    d = adapter.action_dim_per_side
    a = jnp.zeros(n_g * d + n_b * d)
    s = jnp.zeros(adapter.states_dim)
    try:
        adapter.transition(s, a, jax.random.PRNGKey(0))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
