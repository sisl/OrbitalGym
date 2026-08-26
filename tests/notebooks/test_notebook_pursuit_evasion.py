"""Notebook-core-path test for examples/games/pursuit_evasion.ipynb."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from orbitalgym import OrbitalGymEnv, rollout  # noqa: E402
from orbitalgym.env.types import BySide  # noqa: E402
from orbitalgym.games import make_pursuit_evasion  # noqa: E402
from orbitalgym.policies import ZeroControl  # noqa: E402
from orbitalgym.viz import (  # noqa: E402
    plot_reward_diagnostic,
    plot_reward_position_sweep,
)


def _zero_init(c, s, k):
    return None


def test_pe_notebook_core_path():
    cfg = make_pursuit_evasion(
        capture_distance_m=10.0,
        n_guards=1,
        n_bandits=1,
        dt=10.0,
        max_horizon_s=200.0,
        seed=0,
    )
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, jax.random.PRNGKey(42), n_steps=10)

    fig1 = plot_reward_diagnostic(traj, cfg)
    plt.close(fig1)
    fig2 = plot_reward_position_sweep(cfg, grid_steps=8)
    plt.close(fig2)
