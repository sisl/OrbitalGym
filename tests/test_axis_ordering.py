"""Verifies the BATCH → TIME → VEHICLE → FEATURE axis convention."""

from __future__ import annotations

import jax

from examples.reference_scenario import build_config
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import BySide
from orbital_game.policies.library import ZeroControl
from orbital_game.rollout import rollout


def _init_none(c, s, k):
    del c, s, k
    return None


def test_trajectory_axis_order():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    g_pol = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
    b_pol = ZeroControl(n_vehicles=cfg.n_bandits, action_dim=3)

    traj = rollout(
        env,
        BySide(guard=g_pol, bandit=b_pol),
        BySide(guard=_init_none, bandit=_init_none),
        jax.random.PRNGKey(0),
        n_steps=cfg.max_steps,
    )
    n_t = cfg.max_steps
    # Per-side action: (T, N_side, action_dim)
    assert traj.sides.guard.action.shape == (n_t, cfg.n_guards, 3)
    assert traj.sides.bandit.action.shape == (n_t, cfg.n_bandits, 3)
    # Per-side obs (PER_SIDE scope): (T, obs_dim) — no vehicle axis
    assert traj.sides.guard.obs.ndim == 2
    assert traj.sides.bandit.obs.ndim == 2
    # Per-side reward (PER_SIDE scope): (T,) — scalar per step
    assert traj.sides.guard.reward.shape == (n_t,)
    assert traj.sides.bandit.reward.shape == (n_t,)
    # episode_done: (T,) — always 1D
    assert traj.episode_done.shape == (n_t,)


def test_trajectory_vmap_seeds():
    """vmap over a batch of seeds adds a leading B axis to every leaf."""
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    g_pol = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
    b_pol = ZeroControl(n_vehicles=cfg.n_bandits, action_dim=3)

    def _run(key):
        return rollout(
            env,
            BySide(guard=g_pol, bandit=b_pol),
            BySide(guard=_init_none, bandit=_init_none),
            key,
            n_steps=cfg.max_steps,
        )

    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    batched = jax.vmap(_run)(keys)
    n_t = cfg.max_steps
    assert batched.sides.guard.action.shape == (4, n_t, cfg.n_guards, 3)
    assert batched.episode_done.shape == (4, n_t)
