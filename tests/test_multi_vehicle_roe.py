"""Integration test for the multi-vehicle ROE example."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from examples.multi_vehicle_roe import build_config
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import BySide
from orbital_game.rollout import rollout


def _make_policies(n_guards: int, n_bandits: int) -> tuple[BySide, BySide]:
    def guard_policy(ps, obs, key, t):
        del obs, key, t
        return jnp.zeros((n_guards, 3)), ps

    def bandit_policy(ps, obs, key, t):
        del obs, key, t
        return jnp.zeros((n_bandits, 3)), ps

    def init_none(config, env_state, key):
        del config, env_state, key
        return None

    return (
        BySide(guard=guard_policy, bandit=bandit_policy),
        BySide(guard=init_none, bandit=init_none),
    )


def test_multi_vehicle_roe_runs_under_vmap():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    policies, init_fns = _make_policies(cfg.n_guards, cfg.n_bandits)

    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    batched = jax.vmap(lambda k: rollout(env, policies, init_fns, k, n_steps=10))
    traj = batched(keys)
    # Trajectory shape sanity
    assert traj.env_state.ic_valid.shape == (8, 10)
    # Most lanes should produce valid ICs (sep=50m is permissive given ~500m ellipses)
    valid_rate = float(jnp.mean(traj.env_state.ic_valid[:, 0]))
    assert valid_rate > 0.90, f"valid_rate={valid_rate}"
