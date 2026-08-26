"""Tests for viz/reward_diagnostics.py — game-aware reward overlays."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from orbitalgym.env.core import OrbitalGymEnv  # noqa: E402
from orbitalgym.env.types import BySide  # noqa: E402
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard  # noqa: E402
from orbitalgym.games.pursuit_evasion import make_pursuit_evasion  # noqa: E402
from orbitalgym.policies import ZeroControl  # noqa: E402
from orbitalgym.rollout import rollout  # noqa: E402
from orbitalgym.viz.reward_diagnostics import (  # noqa: E402
    plot_lady_bandit_guard_diagnostic,
    plot_pursuit_evasion_diagnostic,
    plot_reward_diagnostic,
)


def _zero_init(c, s, k):
    return None


def test_pe_diagnostic_two_panels(key):
    cfg = make_pursuit_evasion(dt=10.0, max_horizon_s=200.0, capture_distance_m=10.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, key, n_steps=15)
    fig = plot_pursuit_evasion_diagnostic(traj, capture_distance_m=10.0)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_lbg_diagnostic_two_panels_with_threshold(key):
    cfg = make_lady_bandit_guard(
        n_guards=2, n_bandits=1, dt=10.0, max_horizon_s=200.0, breach_radius_m=50.0
    )
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=2),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, key, n_steps=15)
    fig = plot_lady_bandit_guard_diagnostic(traj, breach_radius_m=50.0)
    assert len(fig.axes) == 2
    # Top axes should have a horizontal threshold line.
    ax_top = fig.axes[0]
    has_hline = any(line.get_linestyle() == "--" for line in ax_top.get_lines())
    assert has_hline
    plt.close(fig)


def test_reward_diagnostic_dispatches_pe(key):
    cfg = make_pursuit_evasion(dt=10.0, max_horizon_s=200.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, key, n_steps=15)
    fig = plot_reward_diagnostic(traj, cfg)
    # PE → 2 axes
    assert len(fig.axes) == 2
    plt.close(fig)


def test_reward_diagnostic_dispatches_lbg(key):
    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1, dt=10.0, max_horizon_s=200.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, key, n_steps=15)
    fig = plot_reward_diagnostic(traj, cfg)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_sb_diagnostic_three_panels(key):
    from orbitalgym.games.sun_blocking import make_sun_blocking
    from orbitalgym.viz.reward_diagnostics import plot_sun_blocking_diagnostic

    cfg = make_sun_blocking(dt=10.0, max_horizon_s=200.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, key, n_steps=15)
    fig = plot_sun_blocking_diagnostic(traj, cfg)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_ob_diagnostic_four_panels(key):
    from orbitalgym.games.observation_blocking import make_observation_blocking
    from orbitalgym.viz.reward_diagnostics import plot_observation_blocking_diagnostic

    cfg = make_observation_blocking(dt=10.0, max_horizon_s=200.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, key, n_steps=15)
    fig = plot_observation_blocking_diagnostic(traj, cfg)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_reward_diagnostic_dispatches_sb(key):
    from orbitalgym.games.sun_blocking import make_sun_blocking
    from orbitalgym.viz.reward_diagnostics import plot_reward_diagnostic

    cfg = make_sun_blocking(dt=10.0, max_horizon_s=200.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, key, n_steps=15)
    fig = plot_reward_diagnostic(traj, cfg)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_reward_diagnostic_dispatches_ob(key):
    from orbitalgym.games.observation_blocking import make_observation_blocking
    from orbitalgym.viz.reward_diagnostics import plot_reward_diagnostic

    cfg = make_observation_blocking(dt=10.0, max_horizon_s=200.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=_zero_init, bandit=_zero_init)
    traj = rollout(env, policies, init_ps, key, n_steps=15)
    fig = plot_reward_diagnostic(traj, cfg)
    assert len(fig.axes) == 4
    plt.close(fig)
