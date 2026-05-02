"""3D glyph primitives for animations and per-frame scene rendering.

Each helper draws one visual element on a matplotlib axes:

- ``draw_cube`` — translucent agent cube oriented by a 3x3 rotation
- ``draw_sphere`` — translucent sensor-range sphere (3D)
- ``draw_circle`` — sensor-range circle (2D, for RT/RN/TN plots)
- ``draw_thrust_arrow`` — Δv quiver in RTN
- ``draw_body_axes`` — RGB body-frame axes
- ``draw_belief_ellipsoid`` — 2σ position-uncertainty ellipsoid from a 3x3 covariance

Conventions match ``refs/visualization.py``: cubes via ``Poly3DCollection``,
surfaces via ``ax.plot_surface``. Inputs are numpy arrays (callers convert
from JAX at the boundary).

All helpers return the matplotlib artist they created so callers can update
or remove them across frames if needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_CUBE_VERTICES = (
    np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    * 0.5
)

_CUBE_FACE_INDICES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 3, 7, 4),
    (1, 2, 6, 5),
)


def draw_cube(
    ax: Any,
    center: np.ndarray,
    rotation: np.ndarray | None = None,
    scale: float = 1.0,
    face_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 0.3),
    edge_color: str = "gray",
    linewidth: float = 0.5,
) -> Poly3DCollection:
    """Draw a translucent cube centered at ``center`` with side length ``scale``.

    ``rotation`` is a 3x3 body-to-reference rotation matrix; pass ``None`` for
    an axis-aligned (RTN-aligned) cube. Returns the ``Poly3DCollection`` so
    callers can adjust or remove it.
    """
    R = np.eye(3) if rotation is None else np.asarray(rotation)  # noqa: N806
    rotated = (R @ (_CUBE_VERTICES.T * scale)).T + np.asarray(center)
    faces = [[rotated[j] for j in face] for face in _CUBE_FACE_INDICES]
    poly = Poly3DCollection(
        faces,
        facecolors=[face_color] * 6,
        edgecolors=edge_color,
        linewidths=linewidth,
    )
    ax.add_collection3d(poly)
    return poly


def draw_body_axes(
    ax: Any,
    center: np.ndarray,
    rotation: np.ndarray,
    scale: float = 1.0,
    label: bool = False,
) -> list[Any]:
    """Draw RGB body-frame axes (X red, Y green, Z blue) emanating from ``center``."""
    R = np.asarray(rotation)  # noqa: N806
    c = np.asarray(center)
    artists = []
    colors = ("red", "green", "blue")
    names = ("X", "Y", "Z")
    for i in range(3):
        d = R[:, i] * scale
        q = ax.quiver(*c, *d, color=colors[i], linewidth=1.5, arrow_length_ratio=0.1)
        artists.append(q)
        if label:
            tip = c + d * 1.1
            artists.append(ax.text(*tip, names[i], color=colors[i], fontsize=7))
    return artists


def draw_sphere(
    ax: Any,
    center: np.ndarray,
    radius: float,
    color: str = "gray",
    alpha: float = 0.12,
    n_lat: int = 16,
    n_lon: int = 24,
) -> Any:
    """Draw a translucent sphere — used for sensor-range gating in 3D RTN plots."""
    u = np.linspace(0.0, 2.0 * np.pi, n_lon)
    v = np.linspace(0.0, np.pi, n_lat)
    cx, cy, cz = np.asarray(center)
    x = cx + radius * np.outer(np.cos(u), np.sin(v))
    y = cy + radius * np.outer(np.sin(u), np.sin(v))
    z = cz + radius * np.outer(np.ones_like(u), np.cos(v))
    return ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False)


def draw_circle(
    ax: Any,
    center: np.ndarray,
    radius: float,
    color: str = "gray",
    alpha: float = 0.15,
    n_pts: int = 64,
    fill: bool = True,
) -> Any:
    """Draw a 2D circle — sensor-range overlay for RT/RN/TN projections."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    cx, cy = float(center[0]), float(center[1])
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)
    if fill:
        return ax.fill(xs, ys, color=color, alpha=alpha, linewidth=0)[0]
    return ax.plot(xs, ys, color=color, alpha=alpha)[0]


def draw_thrust_arrow(
    ax: Any,
    center: np.ndarray,
    dv: np.ndarray,
    scale: float = 1.0,
    color: str = "orange",
    linewidth: float = 1.5,
    arrow_length_ratio: float = 0.2,
) -> Any:
    """Draw a Δv arrow at ``center`` pointing in the direction of ``dv``.

    ``scale`` multiplies the dv magnitude. Convention used by the per-rollout
    and animation modules: ``scale = (max_axis_extent / 20) / max_dv_magnitude``,
    so the largest Δv in the rollout occupies 1/20 of the largest axis extent.
    Smaller maneuvers shrink proportionally. Returns the matplotlib quiver
    artist, or ``None`` if ``dv`` is zero.
    """
    d = np.asarray(dv) * scale
    if not np.any(np.isfinite(d)) or np.linalg.norm(d) < 1e-12:
        return None
    return ax.quiver(
        *np.asarray(center),
        *d,
        color=color,
        linewidth=linewidth,
        arrow_length_ratio=arrow_length_ratio,
    )


def draw_belief_ellipsoid(
    ax: Any,
    mean: np.ndarray,
    cov_3x3: np.ndarray,
    color: Any = "tab:blue",
    alpha: float = 0.15,
    sigma: float = 1.0,
    n_lat: int = 12,
    n_lon: int = 18,
) -> Any | None:
    """Draw a Gaussian uncertainty ellipsoid for a 3D position belief.

    ``cov_3x3`` is the 3x3 position covariance (slice ``cov[:3, :3]`` from a
    larger-d EKF cov). ``sigma`` is the ellipsoid scale (default 1 → 1σ).
    Returns ``None`` if cov is non-positive (e.g. all-zero), so the caller
    can skip drawing without special-casing.
    """
    cov = np.asarray(cov_3x3)
    if not np.all(np.isfinite(cov)) or np.linalg.norm(cov) < 1e-18:
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    if np.all(eigvals < 1e-18):
        return None
    radii = sigma * np.sqrt(eigvals)

    u = np.linspace(0.0, 2.0 * np.pi, n_lon)
    v = np.linspace(0.0, np.pi, n_lat)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([sx * radii[0], sy * radii[1], sz * radii[2]], axis=-1)
    pts = pts @ eigvecs.T + np.asarray(mean)
    return ax.plot_surface(
        pts[..., 0],
        pts[..., 1],
        pts[..., 2],
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=False,
    )


def draw_belief_ellipse_2d(
    ax: Any,
    mean_xy: np.ndarray,
    cov_2x2: np.ndarray,
    color: Any = "tab:blue",
    alpha: float = 0.2,
    sigma: float = 1.0,
    n_pts: int = 64,
) -> Any | None:
    """2D analogue of ``draw_belief_ellipsoid`` for RT/RN/TN projections.

    ``sigma`` defaults to 1 (1σ contour). Returns ``None`` for zero or
    non-finite covariance so callers can skip without special-casing.
    """
    cov = np.asarray(cov_2x2)
    if not np.all(np.isfinite(cov)) or np.linalg.norm(cov) < 1e-18:
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    if np.all(eigvals < 1e-18):
        return None
    radii = sigma * np.sqrt(eigvals)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    pts = unit * radii  # (n_pts, 2)
    pts = pts @ eigvecs.T + np.asarray(mean_xy)
    return ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=alpha, linewidth=0)[0]


def quat_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert (w, x, y, z) quaternion to a 3x3 body-to-reference rotation matrix.

    Matches the convention used by the ``Attitude`` state component
    (identity quaternion = (1, 0, 0, 0)).
    """
    q = np.asarray(quat_wxyz, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
