"""Notebook-core-path test for examples/games/observation_blocking.ipynb."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import replace  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from orbital_game import OrbitalGameEnv, rollout  # noqa: E402
from orbital_game.env.types import BySide  # noqa: E402
from orbital_game.games import make_observation_blocking  # noqa: E402
from orbital_game.policies.library import ZeroControl  # noqa: E402
from orbital_game.sampling.side import RelativeEllipse  # noqa: E402
from orbital_game.sampling.spec import ICSpec  # noqa: E402
from orbital_game.viz import (  # noqa: E402
    plot_reward_diagnostic,
    plot_reward_position_sweep,
)


def _zero_init(c, s, k):
    return None


def test_ob_notebook_core_path():
    cfg = make_observation_blocking(
        target_lat_deg=0.0,
        target_lon_deg=0.0,
        min_elevation_deg=5.0,
        target_viewing_distance_m=500.0,
        range_decay_coef=4.0e-6,
        n_guards=1,
        n_bandits=1,
        dt=10.0,
        max_horizon_s=200.0,
        seed=0,
    )
    cfg = replace(
        cfg,
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=500.0,
                phase_rad=0.0,
                sigma_radial_ellipse_m=10.0,
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=600.0,
                phase_rad=jnp.pi / 4.0,
                sigma_radial_ellipse_m=10.0,
            ),
            validators=(),
            max_attempts=100,
        ),
    )
    env = OrbitalGameEnv(cfg)
    policies = BySide(
        guard=ZeroControl(n_vehicles=cfg.n_guards, action_dim=3),
        bandit=ZeroControl(n_vehicles=cfg.n_bandits, action_dim=3),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, jax.random.PRNGKey(42), n_steps=10)

    fig1 = plot_reward_diagnostic(traj, cfg)
    plt.close(fig1)
    fig2 = plot_reward_position_sweep(cfg, grid_steps=8, t_seconds=700.0)
    plt.close(fig2)
