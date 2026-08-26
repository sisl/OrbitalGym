"""Tests for viz/trajectories.py — plots draw without error and populate Axes."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from orbitalgym.viz.trajectories import plot_rn, plot_rt, plot_rtn_3d, plot_tn


def _fake_rtn_trajectory(n_steps: int = 50):
    """Shape (T, N_guards=1, 6) — RTN state with position components
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


def test_plot_rtn_3d_returns_axes_with_data():
    traj = _fake_rtn_trajectory()
    ax = plot_rtn_3d(traj)
    assert ax is not None
    assert ax.has_data()
    plt.close(ax.figure)


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
    # Each vehicle contributes one Line2D — endpoint markers use scatter so
    # they don't appear here.
    assert len(ax.get_lines()) == 2
    plt.close(fig)


def test_plot_rt_endpoints_default_on():
    """By default, each vehicle gets a start (o) and end (x) marker, plotted
    via scatter so collections — not lines — accumulate on the axes."""
    traj = _fake_rtn_trajectory()
    fig, ax = plt.subplots()
    plot_rt(traj, ax=ax)
    # 1 vehicle → 2 scatter calls (o at start, x at end)
    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_rt_endpoints_can_be_disabled():
    traj = _fake_rtn_trajectory()
    fig, ax = plt.subplots()
    plot_rt(traj, ax=ax, show_endpoints=False)
    assert len(ax.collections) == 0
    plt.close(fig)


def test_plot_rt_endpoint_color_matches_line():
    """Marker color must match its trajectory's line color (so the user can
    visually associate a start/end pair with its rollout when multiple
    overlay)."""
    traj = _fake_rtn_trajectory()
    fig, ax = plt.subplots()
    plot_rt(traj, ax=ax)
    line_color = ax.get_lines()[0].get_color()
    # Both scatter collections share that color (matplotlib stores it as RGBA).
    import matplotlib.colors as mcolors

    expected_rgba = mcolors.to_rgba(line_color)
    for coll in ax.collections:
        actual_rgba = (
            tuple(coll.get_facecolor()[0])
            if len(coll.get_facecolor())
            else (tuple(coll.get_edgecolor()[0]))
        )
        assert actual_rgba == expected_rgba
    plt.close(fig)


def test_plot_rt_endpoints_excluded_from_legend():
    """Endpoint markers should not produce extra legend entries — only one per
    vehicle's line trace."""
    t = jnp.linspace(0, 2 * jnp.pi, 20)
    v0 = jnp.stack(
        [t, t * 2, jnp.zeros_like(t), jnp.zeros_like(t), jnp.zeros_like(t), jnp.zeros_like(t)],
        axis=-1,
    )
    traj = v0[:, None, :]  # (T, 1, 6)

    fig, ax = plt.subplots()
    plot_rt(traj, ax=ax)
    legend = ax.legend()
    # Exactly one legend entry (the line "v0"), not three.
    assert len(legend.get_texts()) == 1
    plt.close(fig)


def test_plot_rtn_3d_endpoints_default_on():
    traj = _fake_rtn_trajectory()
    ax = plot_rtn_3d(traj)
    # 1 vehicle → 2 scatter calls in 3D too
    assert len(ax.collections) == 2
    plt.close(ax.figure)


def test_plot_rtn_3d_endpoints_can_be_disabled():
    traj = _fake_rtn_trajectory()
    ax = plot_rtn_3d(traj, show_endpoints=False)
    assert len(ax.collections) == 0
    plt.close(ax.figure)


def test_plot_rt_label_override_for_two_overlaid_rollouts():
    """Pass label= per call so legend() picks up the right names with no
    explicit handle list."""
    traj_a = _fake_rtn_trajectory()
    traj_b = _fake_rtn_trajectory(n_steps=30)

    fig, ax = plt.subplots()
    plot_rt(traj_a, ax=ax, label="rollout 1")
    plot_rt(traj_b, ax=ax, label="rollout 2")

    legend = ax.legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == ["rollout 1", "rollout 2"]
    plt.close(fig)


def test_plot_rt_default_label_unchanged_when_label_omitted():
    traj = _fake_rtn_trajectory()
    fig, ax = plt.subplots()
    plot_rt(traj, ax=ax)  # no label argument
    legend = ax.legend()
    assert [t.get_text() for t in legend.get_texts()] == ["v0"]
    plt.close(fig)


def test_plot_rt_multi_vehicle_label_appends_vehicle_suffix():
    """If you pass label= for a multi-vehicle traj, each vehicle still gets
    its own legend entry distinguished by a v{i} suffix."""
    import jax.numpy as jnp

    t = jnp.linspace(0, 1, 10)
    z = jnp.zeros_like(t)
    v0 = jnp.stack([t, t * 2, z, z, z, z], axis=-1)
    v1 = jnp.stack([t * 3, t * 4, z, z, z, z], axis=-1)
    traj = jnp.stack([v0, v1], axis=1)  # (T, 2, 6)

    fig, ax = plt.subplots()
    plot_rt(traj, ax=ax, label="guard")
    legend = ax.legend()
    assert [t.get_text() for t in legend.get_texts()] == ["guard v0", "guard v1"]
    plt.close(fig)


def test_plot_rtn_3d_accepts_existing_ax_for_overlay():
    """Calling plot_rtn_3d twice with the same ax stacks both rollouts'
    traces and markers in one figure."""
    traj_a = _fake_rtn_trajectory()
    traj_b = _fake_rtn_trajectory(n_steps=30)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_rtn_3d(traj_a, ax=ax, label="rollout 1")
    plot_rtn_3d(traj_b, ax=ax, label="rollout 2")

    # Two lines, four scatter collections (2 endpoints × 2 rollouts)
    assert len(ax.get_lines()) == 2
    assert len(ax.collections) == 4
    legend = ax.legend()
    assert [t.get_text() for t in legend.get_texts()] == ["rollout 1", "rollout 2"]
    plt.close(fig)
