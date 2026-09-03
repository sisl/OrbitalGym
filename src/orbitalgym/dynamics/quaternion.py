"""Unit-quaternion helpers in the (w, x, y, z) convention.

``quat_to_rotation_matrix(q)`` is the body-to-world rotation, matching
``viz.glyphs.quat_to_rotation_matrix`` and the conical sensor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def quat_to_rotation_matrix(q: jax.Array) -> jax.Array:
    """Body-to-world 3x3 rotation for a (w, x, y, z) quaternion.

    A zero quaternion gives identity.
    """
    dtype = q.dtype
    norm = jnp.linalg.norm(q)
    q = q / jnp.maximum(norm, jnp.asarray(1e-12, dtype=dtype))
    w, x, y, z = q[0], q[1], q[2], q[3]
    one = jnp.asarray(1.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    return jnp.stack(
        [
            one - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            one - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            one - two * (x * x + y * y),
        ]
    ).reshape(3, 3)


def quat_multiply(a: jax.Array, b: jax.Array) -> jax.Array:
    """Hamilton product ``a ⊗ b``; applying ``b`` first then ``a``."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return jnp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def axis_angle_to_quat(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Quaternion for a rotation of ``angle`` radians about unit ``axis``."""
    dtype = axis.dtype
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), jnp.asarray(1e-12, dtype=dtype))
    half = jnp.asarray(0.5, dtype=dtype) * angle
    return jnp.concatenate([jnp.sin(half) * axis, jnp.cos(half)[None]])[jnp.array([3, 0, 1, 2])]


def _perpendicular(v: jax.Array) -> jax.Array:
    """A unit vector perpendicular to ``v``."""
    dtype = v.dtype
    candidate = jnp.where(
        jnp.abs(v[0]) < jnp.asarray(0.9, dtype=dtype),
        jnp.array([1.0, 0.0, 0.0], dtype=dtype),
        jnp.array([0.0, 1.0, 0.0], dtype=dtype),
    )
    p = jnp.cross(v, candidate)
    return p / jnp.maximum(jnp.linalg.norm(p), jnp.asarray(1e-12, dtype=dtype))


def rotate_toward(
    quat: jax.Array,
    boresight_body: jax.Array,
    target_dir: jax.Array,
    max_angle: jax.Array | float,
) -> jax.Array:
    """Rotate so the body boresight moves toward ``target_dir`` by at most ``max_angle``.

    The rotation is about the axis perpendicular to the current and target
    directions (world frame), so the boresight follows the shortest arc. An
    antiparallel target rotates about an arbitrary perpendicular axis.
    """
    dtype = quat.dtype
    eps = jnp.asarray(1e-12, dtype=dtype)
    current = quat_to_rotation_matrix(quat) @ boresight_body
    current = current / jnp.maximum(jnp.linalg.norm(current), eps)
    target = target_dir / jnp.maximum(jnp.linalg.norm(target_dir), eps)
    cos_angle = jnp.clip(jnp.dot(current, target), -1.0, 1.0)
    angle = jnp.arccos(cos_angle)
    axis_raw = jnp.cross(current, target)
    axis_norm = jnp.linalg.norm(axis_raw)
    axis = jnp.where(
        axis_norm > 1e-9, axis_raw / jnp.maximum(axis_norm, eps), _perpendicular(current)
    )
    step = jnp.minimum(angle, jnp.asarray(max_angle, dtype=dtype))
    q_delta = axis_angle_to_quat(axis, step)
    q_new = quat_multiply(q_delta, quat)
    return q_new / jnp.linalg.norm(q_new)
