"""Tests for viz/trajectories.py — plots draw without error and populate Axes."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from orbital_game.viz.trajectories import plot_rn, plot_rt, plot_rtn_3d, plot_tn


def _fake_rtn_trajectory(n_steps: int = 50):
    """Shape (T, N_defenders=1, 6) — RTN state with position components
    tracing a known shape so visual tests are deterministic."""
    t = jnp.linspace(0, 2 * jnp.pi, n_steps)
    r = jnp.cos(t) * 100
    th = jnp.sin(t) * 200
    n = jnp.sin(2 * t) * 50
    return jnp.stack(
        [r, th, n, jnp.zeros(n_steps), jnp.zeros(n_steps), jnp.zeros(n_steps)],
        axis=-1,
    )[:, None, :]


def test_plot_rt_returns_axes_with_data():
    traj = _fake_rtn_trajectory()
    fig, ax = plt.subplots()
    plot_rt(traj, ax=ax)
    assert ax.has_data()
    plt.close(fig)


def test_plot_rn_returns_axes_with_data():
    traj = _fake_rtn_trajectory()
    fig, ax = plt.subplots()
    plot_rn(traj, ax=ax)
    assert ax.has_data()
    plt.close(fig)


def test_plot_tn_returns_axes_with_data():
    traj = _fake_rtn_trajectory()
    fig, ax = plt.subplots()
    plot_tn(traj, ax=ax)
    assert ax.has_data()
    plt.close(fig)


def test_plot_rtn_3d_returns_figure():
    traj = _fake_rtn_trajectory()
    fig = plot_rtn_3d(traj)
    assert fig is not None
    plt.close(fig)


def test_plot_rt_handles_multiple_vehicles():
    """With two vehicles, both traces should appear on the same axes."""
    t = jnp.linspace(0, 2 * jnp.pi, 20)
    v0 = jnp.stack(
        [
            jnp.cos(t) * 100,
            jnp.sin(t) * 100,
            jnp.zeros_like(t),
            jnp.zeros_like(t),
            jnp.zeros_like(t),
            jnp.zeros_like(t),
        ],
        axis=-1,
    )
    v1 = jnp.stack(
        [
            jnp.cos(t) * 200,
            jnp.sin(t) * 200,
            jnp.zeros_like(t),
            jnp.zeros_like(t),
            jnp.zeros_like(t),
            jnp.zeros_like(t),
        ],
        axis=-1,
    )
    traj = jnp.stack([v0, v1], axis=1)  # (T, 2, 6)

    fig, ax = plt.subplots()
    plot_rt(traj, ax=ax)
    # Each vehicle contributes one Line2D.
    assert len(ax.get_lines()) == 2
    plt.close(fig)
