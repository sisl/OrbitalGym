"""Integration test for the multi-vehicle ROE example."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from examples.multi_vehicle_roe import build_config
from orbital_game.env.environment import OrbitalGameEnv
from orbital_game.rollout import rollout


def _policy(ps, obs, key, t):
    del obs, key, t
    return jnp.zeros((3, 3)), ps  # 3 defenders, 3D dv


def _init_ps(config, env_state, key):
    del config, env_state, key
    return None


def test_multi_vehicle_roe_runs_under_vmap():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)

    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    batched = jax.vmap(lambda k: rollout(env, _policy, _init_ps, k, n_steps=10))
    traj = batched(keys)
    # Trajectory shape sanity
    assert traj.env_state.ic_valid.shape == (8, 10)
    # Most lanes should produce valid ICs (sep=50m is permissive given ~500m ellipses)
    valid_rate = float(jnp.mean(traj.env_state.ic_valid[:, 0]))
    assert valid_rate > 0.90, f"valid_rate={valid_rate}"
