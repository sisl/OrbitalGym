"""Role-swap invariance: with mirrored ICs and zero policies on both sides,
the trajectory's state evolution doesn't depend on which side is 'controlled'.

For the bootstrap reference scenario (DistanceToReferenceOrbit which only
gives reward to GUARD), the swap test focuses on geometric symmetry: when
both sides are ZeroControl, swapping `controlled_side` must not change the
state dynamics (since both sides are scripted to zero).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from examples.reference_scenario import build_config
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import BySide, Side
from orbital_game.policies import ZeroControl
from orbital_game.rollout import rollout


def test_zero_zero_rollout_is_invariant_to_controlled_side():
    """If both sides have ZeroControl, the trajectory doesn't depend on
    which side is 'controlled' — the symmetric core treats both identically."""
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    g_pol = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
    b_pol = ZeroControl(n_vehicles=cfg.n_bandits, action_dim=3)

    def init_none(c, s, k):
        del c, s, k
        return None

    traj_a = rollout(
        env,
        BySide(guard=g_pol, bandit=b_pol),
        BySide(guard=init_none, bandit=init_none),
        jax.random.PRNGKey(cfg.seed),
        n_steps=cfg.max_steps,
    )

    cfg_swap = dataclasses.replace(cfg, controlled_side=Side.BANDIT)
    env_swap = OrbitalGameEnv(cfg_swap)
    traj_b = rollout(
        env_swap,
        BySide(guard=g_pol, bandit=b_pol),
        BySide(guard=init_none, bandit=init_none),
        jax.random.PRNGKey(cfg_swap.seed),
        n_steps=cfg_swap.max_steps,
    )

    # Numerical state evolution is identical (both sides ZeroControl, same ICs).
    assert jnp.allclose(traj_a.env_state.guards.rtn, traj_b.env_state.guards.rtn)
    assert jnp.allclose(traj_a.env_state.bandits.rtn, traj_b.env_state.bandits.rtn)
    # Trajectory's controlled_side metadata reflects the config.
    assert traj_a.controlled_side is Side.GUARD
    assert traj_b.controlled_side is Side.BANDIT
