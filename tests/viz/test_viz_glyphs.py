"""Tests for viz/glyphs.py — 3D primitives.

These are smoke tests: each helper draws onto an axes and we confirm the
returned artist is non-None and added to the axes. The visual correctness
is verified by exercising the helpers from the animation tests.
"""

import math

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _Poly3DCollection  # noqa: E402

from orbital_game.viz.glyphs import (  # noqa: E402
    draw_belief_ellipse_2d,
    draw_belief_ellipsoid,
    draw_body_axes,
    draw_circle,
    draw_cone_3d,
    draw_cube,
    draw_sphere,
    draw_thrust_arrow,
    draw_wedge_2d,
    quat_to_rotation_matrix,
)


def _ax3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def test_draw_cube_returns_poly3d_collection():
    fig, ax = _ax3d()
    poly = draw_cube(ax, np.array([1.0, 2.0, 3.0]), scale=0.5)
    assert isinstance(poly, Poly3DCollection)
    plt.close(fig)


def test_draw_cube_with_rotation_does_not_change_type():
    fig, ax = _ax3d()
    R = quat_to_rotation_matrix(np.array([0.7071, 0.0, 0.7071, 0.0]))  # noqa: N806
    poly = draw_cube(ax, np.zeros(3), rotation=R, scale=1.0)
    assert isinstance(poly, Poly3DCollection)
    plt.close(fig)


def test_draw_sphere_adds_surface():
    fig, ax = _ax3d()
    surf = draw_sphere(ax, np.zeros(3), radius=2.0)
    assert surf is not None
    plt.close(fig)


def test_draw_circle_returns_artist():
    fig, ax = plt.subplots()
    art = draw_circle(ax, np.array([1.0, 2.0]), radius=0.5)
    assert art is not None
    plt.close(fig)


def test_draw_thrust_arrow_zero_returns_none():
    fig, ax = _ax3d()
    out = draw_thrust_arrow(ax, np.zeros(3), np.zeros(3))
    assert out is None
    plt.close(fig)


def test_draw_thrust_arrow_nonzero_returns_quiver():
    fig, ax = _ax3d()
    out = draw_thrust_arrow(ax, np.zeros(3), np.array([1.0, 0.0, 0.0]), scale=2.0)
    assert out is not None
    plt.close(fig)


def test_draw_body_axes_returns_three_arrows():
    fig, ax = _ax3d()
    artists = draw_body_axes(ax, np.zeros(3), np.eye(3), scale=1.0)
    # 3 quivers (no labels by default)
    assert len(artists) == 3
    plt.close(fig)


def test_draw_belief_ellipsoid_skips_zero_cov():
    fig, ax = _ax3d()
    out = draw_belief_ellipsoid(ax, np.zeros(3), np.zeros((3, 3)))
    assert out is None
    plt.close(fig)


def test_draw_belief_ellipsoid_renders_with_pd_cov():
    fig, ax = _ax3d()
    cov = np.diag([1.0, 4.0, 9.0])
    out = draw_belief_ellipsoid(ax, np.zeros(3), cov, sigma=2.0)
    assert out is not None
    plt.close(fig)


def test_draw_belief_ellipse_2d_renders_with_pd_cov():
    fig, ax = plt.subplots()
    out = draw_belief_ellipse_2d(ax, np.zeros(2), np.diag([1.0, 4.0]), sigma=2.0)
    assert out is not None
    plt.close(fig)


def test_quat_to_rotation_matrix_identity():
    R = quat_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))  # noqa: N806
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_quat_to_rotation_matrix_normalizes():
    # Non-unit quaternion — should still produce a valid rotation matrix.
    q = np.array([2.0, 0.0, 0.0, 0.0])  # 2*identity
    R = quat_to_rotation_matrix(q)  # noqa: N806
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_quat_to_rotation_matrix_zero_returns_identity():
    R = quat_to_rotation_matrix(np.zeros(4))  # noqa: N806
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_draw_wedge_2d_returns_wedge_patch():
    fig, ax = plt.subplots()
    patch = draw_wedge_2d(
        ax,
        apex=jnp.array([0.0, 0.0], dtype=jnp.float32),
        axis_xy=jnp.array([1.0, 0.0], dtype=jnp.float32),
        half_angle_rad=math.radians(30.0),
        length=10.0,
    )
    assert isinstance(patch, Wedge)
    plt.close(fig)


def test_draw_cone_3d_returns_poly3d_collection():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    coll = draw_cone_3d(
        ax,
        apex=jnp.zeros(3, dtype=jnp.float32),
        axis=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        half_angle_rad=math.radians(30.0),
        length=10.0,
    )
    assert isinstance(coll, _Poly3DCollection)
    plt.close(fig)
