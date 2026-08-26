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
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide, Side
from orbitalgym.policies import ZeroControl
from orbitalgym.rollout import rollout


def test_zero_zero_rollout_is_invariant_to_controlled_side():
    """If both sides have ZeroControl, the trajectory doesn't depend on
    which side is 'controlled' — the symmetric core treats both identically."""
    cfg = build_config()
    env = OrbitalGymEnv(cfg)
    g_pol = ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards)
    b_pol = ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits)

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
    env_swap = OrbitalGymEnv(cfg_swap)
    g_pol_swap = ZeroControl(command_cls=env_swap.guard_command_cls, n_vehicles=cfg_swap.n_guards)
    b_pol_swap = ZeroControl(command_cls=env_swap.bandit_command_cls, n_vehicles=cfg_swap.n_bandits)
    traj_b = rollout(
        env_swap,
        BySide(guard=g_pol_swap, bandit=b_pol_swap),
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
