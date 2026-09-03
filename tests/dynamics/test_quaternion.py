"""Quaternion helpers shared by pointing, cones, and links."""

import jax.numpy as jnp

from orbitalgym.dynamics.quaternion import (
    axis_angle_to_quat,
    quat_multiply,
    quat_to_rotation_matrix,
    rotate_toward,
)

X = jnp.array([1.0, 0.0, 0.0])
Y = jnp.array([0.0, 1.0, 0.0])
Z = jnp.array([0.0, 0.0, 1.0])
IDENTITY = jnp.array([1.0, 0.0, 0.0, 0.0])


def _angle_between(a, b):
    return jnp.arccos(jnp.clip(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)), -1, 1))


def test_axis_angle_quarter_turn_about_z_maps_x_to_y():
    q = axis_angle_to_quat(Z, jnp.pi / 2)
    assert jnp.allclose(quat_to_rotation_matrix(q) @ X, Y, atol=1e-6)


def test_quat_multiply_composes_rotations():
    q1 = axis_angle_to_quat(Z, jnp.pi / 2)
    q2 = axis_angle_to_quat(Z, jnp.pi / 2)
    q = quat_multiply(q2, q1)
    assert jnp.allclose(quat_to_rotation_matrix(q) @ X, -X, atol=1e-6)


def test_rotate_toward_moves_by_at_most_max_angle():
    q = rotate_toward(IDENTITY, X, Y, max_angle=0.1)
    boresight = quat_to_rotation_matrix(q) @ X
    assert jnp.allclose(_angle_between(boresight, X), 0.1, atol=1e-6)
    assert jnp.allclose(_angle_between(boresight, Y), jnp.pi / 2 - 0.1, atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(q), 1.0, atol=1e-6)


def test_rotate_toward_reaches_target_when_within_limit():
    q = rotate_toward(IDENTITY, X, Y, max_angle=2.0)
    assert jnp.allclose(quat_to_rotation_matrix(q) @ X, Y, atol=1e-6)


def test_rotate_toward_handles_antiparallel_target():
    q = rotate_toward(IDENTITY, X, -X, max_angle=0.5)
    boresight = quat_to_rotation_matrix(q) @ X
    assert jnp.allclose(_angle_between(boresight, X), 0.5, atol=1e-6)


def test_rotate_toward_is_identity_when_aligned():
    q = rotate_toward(IDENTITY, X, X, max_angle=0.5)
    assert jnp.allclose(q, IDENTITY, atol=1e-6)
