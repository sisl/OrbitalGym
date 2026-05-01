"""Trajectory plots for the RT, RN, TN planes and the full 3D RTN view.

All functions accept a `(T, N, >=3)` array — T time steps, N vehicles, state
vector where the first three components are position. Works for both RTN
(6-dim) and RT (4-dim) states since only the position columns are indexed.

Axis convention (orbital-mechanics standard):
  - R (radial) is always the y-axis (in plot_rt and plot_rn)
  - T (along-track) is always the x-axis (in plot_rt and plot_tn)
  - N (cross-track) is the remaining axis as needed (x in plot_rn, y in plot_tn)

Endpoint markers: each plot draws an `o` at the trajectory start and an `x`
at the end, in the line's color. Markers are drawn via `ax.scatter` (not
`ax.plot`) so they don't show up in `ax.get_lines()` or the legend. Pass
`show_endpoints=False` to suppress them.

Per-trace labels: pass `label="rollout 1"` (or any string) to override the
default `v{i}` label. Then `ax.legend()` (with no args) picks up exactly the
line labels — no spurious entries from scatter markers. With multi-vehicle
trajectories, `label` is used as a prefix and the per-vehicle suffix `" v{i}"`
is appended automatically.

Note: For RT states (shape `(T, N, 4)`), only plot_rt is meaningful — calling
plot_rn/plot_tn/plot_rtn_3d on pure RT trajectories will index column 2 (the
xdot velocity component) as if it were N, producing incorrect plots.
"""

from __future__ import annotations

from typing import Any

import jax
import matplotlib.pyplot as plt


def _draw_endpoints_2d(ax: plt.Axes, xs: jax.Array, ys: jax.Array, color: Any) -> None:
    """Place start (o) and end (x) markers on a 2D axes in the given color.

    Uses scatter (not plot) and an explicit `label="_nolegend_"` so the markers
    do not appear in `ax.get_lines()` and are explicitly excluded from
    matplotlib's legend handle collection — even by a positional call like
    `ax.legend(["A", "B"])`. `color` accepts any matplotlib color spec.
    """
    ax.scatter(xs[0], ys[0], marker="o", color=color, zorder=3, label="_nolegend_")
    ax.scatter(xs[-1], ys[-1], marker="x", color=color, zorder=3, label="_nolegend_")


def _draw_endpoints_3d(ax, xs: jax.Array, ys: jax.Array, zs: jax.Array, color: Any) -> None:
    """3D analogue of `_draw_endpoints_2d` (also explicitly `_nolegend_`)."""
    ax.scatter(xs[0], ys[0], zs[0], marker="o", color=color, depthshade=False, label="_nolegend_")
    ax.scatter(
        xs[-1], ys[-1], zs[-1], marker="x", color=color, depthshade=False, label="_nolegend_"
    )


def _label_for(label: str | None, i: int, n_vehicles: int) -> str:
    """Resolve a line label.

    - If the caller didn't provide a label, fall back to `v{i}`.
    - If they did and there's only one vehicle, use it as-is.
    - If there are multiple vehicles, append the per-vehicle suffix so each
      trace stays distinguishable.
    """
    if label is None:
        return f"v{i}"
    if n_vehicles == 1:
        return label
    return f"{label} v{i}"


def plot_rt(
    traj: jax.Array,
    ax: plt.Axes | None = None,
    show_endpoints: bool = True,
    label: str | None = None,
) -> plt.Axes:
    """In-plane plot: T on x-axis, R on y-axis (one trace per vehicle)."""
    if ax is None:
        _, ax = plt.subplots()
    n = traj.shape[1]
    for i in range(n):
        # T on x (col 1), R on y (col 0)
        xs, ys = traj[:, i, 1], traj[:, i, 0]
        (line,) = ax.plot(xs, ys, label=_label_for(label, i, n))
        if show_endpoints:
            _draw_endpoints_2d(ax, xs, ys, line.get_color())
    ax.set_xlabel("T (m)")
    ax.set_ylabel("R (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    return ax


def plot_rn(
    traj: jax.Array,
    ax: plt.Axes | None = None,
    show_endpoints: bool = True,
    label: str | None = None,
) -> plt.Axes:
    """N on x-axis, R on y-axis (one trace per vehicle)."""
    if ax is None:
        _, ax = plt.subplots()
    n = traj.shape[1]
    for i in range(n):
        # N on x (col 2), R on y (col 0)
        xs, ys = traj[:, i, 2], traj[:, i, 0]
        (line,) = ax.plot(xs, ys, label=_label_for(label, i, n))
        if show_endpoints:
            _draw_endpoints_2d(ax, xs, ys, line.get_color())
    ax.set_xlabel("N (m)")
    ax.set_ylabel("R (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    return ax


def plot_tn(
    traj: jax.Array,
    ax: plt.Axes | None = None,
    show_endpoints: bool = True,
    label: str | None = None,
) -> plt.Axes:
    """T on x-axis, N on y-axis (one trace per vehicle)."""
    if ax is None:
        _, ax = plt.subplots()
    n = traj.shape[1]
    for i in range(n):
        # T on x (col 1), N on y (col 2)
        xs, ys = traj[:, i, 1], traj[:, i, 2]
        (line,) = ax.plot(xs, ys, label=_label_for(label, i, n))
        if show_endpoints:
            _draw_endpoints_2d(ax, xs, ys, line.get_color())
    ax.set_xlabel("T (m)")
    ax.set_ylabel("N (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    return ax


def plot_rtn_3d(
    traj: jax.Array,
    ax=None,
    show_endpoints: bool = True,
    label: str | None = None,
):
    """3D RTN view per vehicle.

    If `ax` is None, a new Figure with a 3D subplot is created. Otherwise the
    given `ax` (which must be a 3D-projection axes) is reused — letting you
    overlay multiple rollouts with their own labels and endpoint markers.

    Returns the 3D axes; the figure is reachable via `ax.figure`.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    n = traj.shape[1]
    for i in range(n):
        xs = traj[:, i, 0]
        ys = traj[:, i, 1]
        zs = traj[:, i, 2]
        (line,) = ax.plot(xs, ys, zs, label=_label_for(label, i, n))
        if show_endpoints:
            _draw_endpoints_3d(ax, xs, ys, zs, line.get_color())
    ax.set_xlabel("R (m)")
    ax.set_ylabel("T (m)")
    ax.set_zlabel("N (m)")
    return ax
