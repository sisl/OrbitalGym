"""Tests for viz/per_rollout.py — per-rollout multi-actor static plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from orbitalgym.env.core import OrbitalGymEnv  # noqa: E402
from orbitalgym.env.types import BySide  # noqa: E402
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard  # noqa: E402
from orbitalgym.games.pursuit_evasion import make_pursuit_evasion  # noqa: E402
from orbitalgym.policies import ZeroControl  # noqa: E402
from orbitalgym.rollout import rollout  # noqa: E402
from orbitalgym.viz.per_rollout import (  # noqa: E402
    plot_rollout_3d,
    plot_rollout_mass,
    plot_rollout_panels,
    plot_rollout_rewards,
)


@pytest.fixture
def lbg_traj_2g_2b(key):
    cfg = make_lady_bandit_guard(n_guards=2, n_bandits=2, dt=10.0, max_horizon_s=200.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=2),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=2),
    )
    init_ps = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)
    return rollout(env, policies, init_ps, key, n_steps=15)


@pytest.fixture
def pe_traj_no_mass(key):
    cfg = make_pursuit_evasion(dt=10.0, max_horizon_s=200.0)
    env = OrbitalGymEnv(cfg)
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=1),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=1),
    )
    init_ps = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)
    return rollout(env, policies, init_ps, key, n_steps=15)


def test_plot_rollout_3d_draws_one_line_per_actor(lbg_traj_2g_2b):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_rollout_3d(lbg_traj_2g_2b, ax=ax)
    # 2 guards + 2 bandits = 4 trajectory lines
    assert len(ax.get_lines()) == 4
    plt.close(fig)


def test_plot_rollout_panels_creates_four_axes(lbg_traj_2g_2b):
    fig = plot_rollout_panels(lbg_traj_2g_2b)
    # 3 2D axes + 1 3D axes
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_rollout_mass_one_line_per_vehicle_with_mass(lbg_traj_2g_2b):
    fig, ax = plt.subplots()
    plot_rollout_mass(lbg_traj_2g_2b, ax=ax)
    # LBG default: 2 guards have Mass, bandits do not → 2 lines
    assert len(ax.get_lines()) == 2
    plt.close(fig)


def test_plot_rollout_mass_raises_when_no_mass_anywhere(pe_traj_no_mass):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Mass component"):
        plot_rollout_mass(pe_traj_no_mass, ax=ax)
    plt.close(fig)


def test_plot_rollout_rewards_handles_per_side_and_per_vehicle(lbg_traj_2g_2b):
    fig, ax = plt.subplots()
    plot_rollout_rewards(lbg_traj_2g_2b, ax=ax)
    assert ax.has_data()
    plt.close(fig)
