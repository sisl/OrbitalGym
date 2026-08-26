"""Tests for viz/summaries.py — reward + mass summary plots."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from orbitalgym.viz.summaries import plot_mass_curve, plot_reward_curve


def test_plot_reward_curve_single_trajectory():
    rewards = jnp.sin(jnp.linspace(0, 2 * jnp.pi, 50))
    fig, ax = plt.subplots()
    plot_reward_curve(rewards, ax=ax)
    assert ax.has_data()
    # 1D input → one line
    assert len(ax.get_lines()) == 1
    plt.close(fig)


def test_plot_reward_curve_batched_overlays_mean():
    rewards = jnp.sin(jnp.linspace(0, 2 * jnp.pi, 50))[None, :] + 0.1 * jnp.arange(4)[:, None]
    fig, ax = plt.subplots()
    plot_reward_curve(rewards, ax=ax)
    assert ax.has_data()
    # 4 faint traces + 1 mean line = 5
    assert len(ax.get_lines()) == 5
    plt.close(fig)


def test_plot_reward_curve_rejects_unsupported_rank():
    rewards = jnp.zeros((2, 3, 4))  # 3D — not supported
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="1D or 2D"):
        plot_reward_curve(rewards, ax=ax)
    plt.close(fig)


def test_plot_mass_curve_per_vehicle_lines():
    # (T, N=2) propellant mass over time
    mass = jnp.linspace(10.0, 0.0, 50)[:, None] * jnp.ones((1, 2))
    fig, ax = plt.subplots()
    plot_mass_curve(mass, ax=ax)
    assert ax.has_data()
    assert len(ax.get_lines()) == 2
    plt.close(fig)
