"""LBG with comms: guard pays a per-broadcast cost; bandit sees the leak."""

import jax
import jax.numpy as jnp

from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.registry import ActionComponentKey


def test_lbg_comms_builder_wires_communicate():
    cfg = make_lady_bandit_guard(with_communication=True, comm_cost=5.0)
    assert ActionComponentKey.COMMUNICATE in cfg.guard_action_components
    assert ActionComponentKey.COMMUNICATE not in cfg.bandit_action_components


def test_lbg_comms_charges_cost_when_active():
    cfg = make_lady_bandit_guard(with_communication=True, comm_cost=5.0)
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))

    silent = env.guard_command_cls.zeros(cfg.n_guards)
    bandit_cmd = env.bandit_command_cls.zeros(cfg.n_bandits)
    out_silent = env.step(
        jax.random.PRNGKey(1), state, Actions(sides=BySide(guard=silent, bandit=bandit_cmd))
    )

    talking = silent.replace(active=jnp.array([True]))
    out_talking = env.step(
        jax.random.PRNGKey(1), state, Actions(sides=BySide(guard=talking, bandit=bandit_cmd))
    )

    diff = out_silent.outputs.guard.reward - out_talking.outputs.guard.reward
    assert jnp.isclose(diff, 5.0)
