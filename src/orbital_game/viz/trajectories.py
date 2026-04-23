"""Trajectory plots for the RT, RN, TN planes and the full 3D RTN view.

All functions accept a `(T, N, >=3)` array — T time steps, N vehicles, state
vector where the first three components are position. Works for both RTN
(6-dim) and RT (4-dim) states since only the position columns are indexed.

Note: For RT states (shape `(T, N, 4)`), only plot_rt is meaningful — calling
plot_rn/plot_tn/plot_rtn_3d on pure RT trajectories will index column 2 (the
xdot velocity component) as if it were N, producing incorrect plots.
"""

from __future__ import annotations

import jax
import matplotlib.pyplot as plt


def plot_rt(traj: jax.Array, ax: plt.Axes | None = None) -> plt.Axes:
    """In-plane R vs T per vehicle."""
    if ax is None:
        _, ax = plt.subplots()
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], label=f"v{i}")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("T (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    return ax


def plot_rn(traj: jax.Array, ax: plt.Axes | None = None) -> plt.Axes:
    """R vs N per vehicle."""
    if ax is None:
        _, ax = plt.subplots()
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 2], label=f"v{i}")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("N (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    return ax


def plot_tn(traj: jax.Array, ax: plt.Axes | None = None) -> plt.Axes:
    """T vs N per vehicle."""
    if ax is None:
        _, ax = plt.subplots()
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 1], traj[:, i, 2], label=f"v{i}")
    ax.set_xlabel("T (m)")
    ax.set_ylabel("N (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    return ax


def plot_rtn_3d(traj: jax.Array) -> plt.Figure:
    """3D RTN view per vehicle. Returns the Figure (not the Axes) because the
    3D projection needs a dedicated subplot and the caller usually wants the
    whole figure to save/show."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], label=f"v{i}")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("T (m)")
    ax.set_zlabel("N (m)")
    return fig
