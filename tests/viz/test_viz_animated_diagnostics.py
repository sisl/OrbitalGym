"""Smoke tests for animated reward diagnostics."""

import matplotlib

matplotlib.use("Agg")

import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from orbital_game import OrbitalGameEnv, rollout  # noqa: E402
from orbital_game.env.types import BySide  # noqa: E402
from orbital_game.games import make_observation_blocking, make_sun_blocking  # noqa: E402
from orbital_game.policies import ZeroControl  # noqa: E402
from orbital_game.viz import (  # noqa: E402
    plot_observation_blocking_animated_diagnostic,
    plot_sun_blocking_animated_diagnostic,
)


def _zero_init(c, s, k):
    return None


def _short_rollout(make_cfg):
    cfg = make_cfg(dt=10.0, max_horizon_s=200.0)
    env = OrbitalGameEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, jax.random.PRNGKey(0), n_steps=15)
    return cfg, traj


def test_sb_animated_diagnostic_renders_frames():
    cfg, traj = _short_rollout(make_sun_blocking)
    fig, anim = plot_sun_blocking_animated_diagnostic(traj, cfg, n_frames=4)
    # Drive the animation's frame function manually to verify each frame renders.
    for frame_idx in anim._iter_gen():
        anim._func(frame_idx)
    plt.close(fig)


def test_ob_animated_diagnostic_renders_frames():
    cfg, traj = _short_rollout(make_observation_blocking)
    fig, anim = plot_observation_blocking_animated_diagnostic(traj, cfg, n_frames=4)
    for frame_idx in anim._iter_gen():
        anim._func(frame_idx)
    plt.close(fig)


def test_sb_animated_diagnostic_returns_figure_and_anim():
    cfg, traj = _short_rollout(make_sun_blocking)
    fig, anim = plot_sun_blocking_animated_diagnostic(traj, cfg, n_frames=2)
    assert fig is not None
    assert anim is not None
    # Two axes expected (3D scene + cumulative reward).
    assert len(fig.axes) == 2
    plt.close(fig)


def test_ob_animated_diagnostic_returns_three_axes():
    cfg, traj = _short_rollout(make_observation_blocking)
    fig, anim = plot_observation_blocking_animated_diagnostic(traj, cfg, n_frames=2)
    assert fig is not None
    assert anim is not None
    # Three axes expected (3D scene + cumulative reward + elevation).
    assert len(fig.axes) == 3
    plt.close(fig)
