"""Tests for belief/rollout.py — BeliefRollout helper."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.belief.kf import KFBeliefUpdater, KFFromTruthInitializer
from orbital_game.belief.rollout import BeliefRollout
from orbital_game.env.types import Actions, BySide
from orbital_game.games.lady_bandit_guard import make_lady_bandit_guard
from orbital_game.observations.range_limited import RangeLimitedObservation


def test_belief_rollout_reset_returns_initial_beliefs():
    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=2)
    layout_obs = RangeLimitedObservation(
        layout=_LayoutAdapter(cfg.n_guards, cfg.n_bandits, 6),
        sensor_range_m=10000.0,
        sigma_range=1.0,
    )
    cfg_with = _replace_obs_fn(cfg, layout_obs)
    env = _make_env(cfg_with)

    init = KFFromTruthInitializer(
        layout=_LayoutAdapter(cfg.n_guards, cfg.n_bandits, 6),
        variance_diag=jnp.ones(6),
    )
    upd = KFBeliefUpdater(
        stm=jnp.eye(6),
        control_matrix=jnp.zeros((6, 3)),
        process_noise=jnp.eye(6) * 0.01,
    )
    rollout = BeliefRollout(
        env=env,
        guard_belief_initializer=init,
        guard_belief_updater=upd,
        bandit_belief_initializer=init,
        bandit_belief_updater=upd,
    )

    state, beliefs = rollout.reset(jax.random.PRNGKey(0))
    assert beliefs.guard.mean.shape == (cfg.n_guards, cfg.n_guards + cfg.n_bandits, 6)
    assert beliefs.bandit.mean.shape == (cfg.n_bandits, cfg.n_guards + cfg.n_bandits, 6)


def test_belief_rollout_step_advances_belief_with_predict_and_correct():
    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=2)
    obs_fn = RangeLimitedObservation(
        layout=_LayoutAdapter(cfg.n_guards, cfg.n_bandits, 6),
        sensor_range_m=1e9,  # everyone in range
        sigma_range=1.0,
    )
    cfg_with = _replace_obs_fn(cfg, obs_fn)
    env = _make_env(cfg_with)

    init = KFFromTruthInitializer(
        layout=_LayoutAdapter(cfg.n_guards, cfg.n_bandits, 6),
        variance_diag=jnp.ones(6) * 5.0,
    )
    upd = KFBeliefUpdater(
        stm=jnp.eye(6),
        control_matrix=jnp.zeros((6, 3)),
        process_noise=jnp.eye(6) * 0.01,
    )
    rollout = BeliefRollout(
        env=env,
        guard_belief_initializer=init,
        guard_belief_updater=upd,
        bandit_belief_initializer=init,
        bandit_belief_updater=upd,
    )

    state, beliefs = rollout.reset(jax.random.PRNGKey(0))
    actions = Actions(
        sides=BySide(
            guard=jnp.zeros((cfg.n_guards, 3)),
            bandit=jnp.zeros((cfg.n_bandits, 3)),
        )
    )
    state2, beliefs2, _step = rollout.step(jax.random.PRNGKey(1), state, beliefs, actions)
    initial_trace = float(jnp.trace(beliefs.guard.cov[0, 1]))
    after_trace = float(jnp.trace(beliefs2.guard.cov[0, 1]))
    assert after_trace < initial_trace


# --- Helpers ---


class _LayoutAdapter:
    def __init__(self, n_guards, n_bandits, d):
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


def _replace_obs_fn(cfg, obs_fn):
    import dataclasses

    return dataclasses.replace(cfg, guard_observation_fn=obs_fn, bandit_observation_fn=obs_fn)


def _make_env(cfg):
    from orbital_game.env.core import OrbitalGameEnv

    return OrbitalGameEnv(cfg)
