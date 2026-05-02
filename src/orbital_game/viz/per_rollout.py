"""Per-rollout multi-actor static plots.

These functions take a single ``Trajectory`` (from ``orbital_game.rollout``)
and produce visualizations that show every guard *and* every bandit in the
scenario together, color-coded by side. Use them when you want to see the
geometry of one specific outcome rather than overlaying many rollouts of one
policy (which is what ``viz.trajectories`` already supports).

Color convention:

- Guards: shades of blue (``plt.cm.Blues``)
- Bandits: shades of red (``plt.cm.Reds``)

Endpoint markers (``o`` start, ``x`` end) follow the same scheme as
``viz.trajectories`` — drawn via ``ax.scatter`` with ``label="_nolegend_"``
so they don't pollute legends.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from orbital_game.env.types import Side, Trajectory


def _side_colors(n: int, side: Side) -> list[Any]:
    """Pick ``n`` distinguishable colors from the side's colormap."""
    cmap = mpl.colormaps["Blues" if side is Side.GUARD else "Reds"]
    if n == 1:
        return [cmap(0.75)]
    return [cmap(0.45 + 0.5 * i / max(n - 1, 1)) for i in range(n)]


def _position_array(side_state: Any) -> np.ndarray:
    """Pull a ``(T, N, 3)`` position array out of a guard/bandit state.

    Supports both RTN (6-dim) and RT (4-dim, padded with zeros for N).
    Returns numpy (caller will hand to matplotlib).
    """
    if hasattr(side_state, "rtn"):
        rtn = np.asarray(side_state.rtn)
        return rtn[..., :3]
    if hasattr(side_state, "rt"):
        rt = np.asarray(side_state.rt)
        rn = np.zeros(rt.shape[:-1] + (1,), dtype=rt.dtype)
        return np.concatenate([rt[..., :2], rn], axis=-1)
    raise AttributeError("Side state has no `rtn` or `rt` field — cannot extract positions.")


def _has_rtn(side_state: Any) -> bool:
    return hasattr(side_state, "rtn")


def _draw_endpoints_3d(ax: Any, xyz: np.ndarray, color: Any) -> None:
    ax.scatter(*xyz[0], marker="o", color=color, depthshade=False, label="_nolegend_")
    ax.scatter(*xyz[-1], marker="x", color=color, depthshade=False, label="_nolegend_")


def _draw_endpoints_2d(ax: Any, xs: np.ndarray, ys: np.ndarray, color: Any) -> None:
    ax.scatter(xs[0], ys[0], marker="o", color=color, zorder=3, label="_nolegend_")
    ax.scatter(xs[-1], ys[-1], marker="x", color=color, zorder=3, label="_nolegend_")


def plot_rollout_3d(
    traj: Trajectory,
    ax: Any | None = None,
    show_endpoints: bool = True,
    legend: bool = True,
) -> Any:
    """Plot all guard + bandit trajectories of a single rollout in 3D RTN."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    g_xyz = _position_array(traj.env_state.guards)  # (T, n_guards, 3)
    b_xyz = _position_array(traj.env_state.bandits)  # (T, n_bandits, 3)

    g_colors = _side_colors(g_xyz.shape[1], Side.GUARD)
    b_colors = _side_colors(b_xyz.shape[1], Side.BANDIT)

    for i in range(g_xyz.shape[1]):
        ax.plot(
            g_xyz[:, i, 0],
            g_xyz[:, i, 1],
            g_xyz[:, i, 2],
            color=g_colors[i],
            label=f"guard {i}",
        )
        if show_endpoints:
            _draw_endpoints_3d(ax, g_xyz[:, i, :], g_colors[i])

    for j in range(b_xyz.shape[1]):
        ax.plot(
            b_xyz[:, j, 0],
            b_xyz[:, j, 1],
            b_xyz[:, j, 2],
            color=b_colors[j],
            label=f"bandit {j}",
        )
        if show_endpoints:
            _draw_endpoints_3d(ax, b_xyz[:, j, :], b_colors[j])

    ax.set_xlabel("R (m)")
    ax.set_ylabel("T (m)")
    ax.set_zlabel("N (m)")
    if legend:
        ax.legend(loc="best", fontsize=8)
    return ax


def _plot_2d_panel(
    ax: Any,
    xs_g: np.ndarray,
    ys_g: np.ndarray,
    xs_b: np.ndarray,
    ys_b: np.ndarray,
    show_endpoints: bool,
) -> None:
    g_colors = _side_colors(xs_g.shape[1], Side.GUARD)
    b_colors = _side_colors(xs_b.shape[1], Side.BANDIT)
    for i in range(xs_g.shape[1]):
        ax.plot(xs_g[:, i], ys_g[:, i], color=g_colors[i], label=f"guard {i}")
        if show_endpoints:
            _draw_endpoints_2d(ax, xs_g[:, i], ys_g[:, i], g_colors[i])
    for j in range(xs_b.shape[1]):
        ax.plot(xs_b[:, j], ys_b[:, j], color=b_colors[j], label=f"bandit {j}")
        if show_endpoints:
            _draw_endpoints_2d(ax, xs_b[:, j], ys_b[:, j], b_colors[j])
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)


def plot_rollout_panels(
    traj: Trajectory,
    fig: Any | None = None,
    show_endpoints: bool = True,
) -> Any:
    """Multi-panel view of a single rollout: RT, RN, TN, and 3D RTN.

    For RT-only scenarios (``side_state.rt`` instead of ``rtn``), only the
    RT panel is populated; RN/TN/3D are skipped.
    """
    g_xyz = _position_array(traj.env_state.guards)
    b_xyz = _position_array(traj.env_state.bandits)
    has_rtn = _has_rtn(traj.env_state.guards) and _has_rtn(traj.env_state.bandits)

    if fig is None:
        fig = plt.figure(figsize=(12, 10))

    if has_rtn:
        ax_rt = fig.add_subplot(2, 2, 1)
        ax_rn = fig.add_subplot(2, 2, 2)
        ax_tn = fig.add_subplot(2, 2, 3)
        ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
    else:
        ax_rt = fig.add_subplot(1, 1, 1)
        ax_rn = ax_tn = ax_3d = None

    _plot_2d_panel(
        ax_rt, g_xyz[..., 1], g_xyz[..., 0], b_xyz[..., 1], b_xyz[..., 0], show_endpoints
    )
    ax_rt.set_xlabel("T (m)")
    ax_rt.set_ylabel("R (m)")
    ax_rt.set_title("RT (in-plane)")
    ax_rt.legend(loc="best", fontsize=8)

    if ax_rn is not None:
        _plot_2d_panel(
            ax_rn, g_xyz[..., 2], g_xyz[..., 0], b_xyz[..., 2], b_xyz[..., 0], show_endpoints
        )
        ax_rn.set_xlabel("N (m)")
        ax_rn.set_ylabel("R (m)")
        ax_rn.set_title("RN")

    if ax_tn is not None:
        _plot_2d_panel(
            ax_tn, g_xyz[..., 1], g_xyz[..., 2], b_xyz[..., 1], b_xyz[..., 2], show_endpoints
        )
        ax_tn.set_xlabel("T (m)")
        ax_tn.set_ylabel("N (m)")
        ax_tn.set_title("TN")

    if ax_3d is not None:
        plot_rollout_3d(traj, ax=ax_3d, show_endpoints=show_endpoints, legend=False)
        ax_3d.set_title("3D RTN")

    fig.tight_layout()
    return fig


def plot_rollout_mass(traj: Trajectory, ax: Any | None = None) -> Any:
    """Plot per-vehicle propellant mass over time, separated by side.

    Quietly skips sides that don't carry the ``Mass`` component. Raises
    ``ValueError`` only if neither side has it.
    """
    if ax is None:
        _, ax = plt.subplots()

    plotted = 0
    for side, side_state in (
        (Side.GUARD, traj.env_state.guards),
        (Side.BANDIT, traj.env_state.bandits),
    ):
        if not hasattr(side_state, "propellant_mass"):
            continue
        m = np.asarray(side_state.propellant_mass)  # (T, N)
        n = m.shape[1]
        colors = _side_colors(n, side)
        for i in range(n):
            ax.plot(m[:, i], color=colors[i], label=f"{side.value} {i}")
        plotted += n

    if plotted == 0:
        raise ValueError("Neither side carries the Mass component — no propellant mass to plot.")

    ax.set_xlabel("step")
    ax.set_ylabel("propellant mass (kg)")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_rollout_rewards(traj: Trajectory, ax: Any | None = None) -> Any:
    """Plot guard and bandit reward curves.

    Per-vehicle reward (shape ``(T, N)``) is plotted as one line per vehicle
    plus a bold mean. Per-side reward (shape ``(T,)``) is plotted as a single
    line.
    """
    if ax is None:
        _, ax = plt.subplots()

    for side, side_traj in (
        (Side.GUARD, traj.sides.guard),
        (Side.BANDIT, traj.sides.bandit),
    ):
        r = jnp.asarray(side_traj.reward)
        if r.ndim == 1:
            color = "tab:blue" if side is Side.GUARD else "tab:red"
            ax.plot(r, color=color, linewidth=2.0, label=side.value)
        elif r.ndim == 2:
            n = r.shape[1]
            colors = _side_colors(n, side)
            for i in range(n):
                ax.plot(r[:, i], color=colors[i], alpha=0.5, label=f"{side.value} {i}")
            mean_color = "tab:blue" if side is Side.GUARD else "tab:red"
            ax.plot(r.mean(axis=1), color=mean_color, linewidth=2.0, label=f"{side.value} mean")
        else:
            raise ValueError(f"reward for side {side.value} has shape {r.shape}; expected 1D or 2D")

    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    return ax
