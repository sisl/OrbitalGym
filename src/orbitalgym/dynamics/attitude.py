"""Rigid-body attitude dynamics.

State: (quat (n, 4), omega (n, 3)) — wxyz convention matching state.components.Attitude.
Control: applied torque (n, 3) in the body frame.

The integrator is RK4 + per-step quaternion renormalization. Per-axis
omega_max clipping after integration models reaction-wheel saturation
(commanded torque past the wall produces no further angular acceleration).

Float-literal handling: this module wraps Python floats (e.g., the 0.5 in
the quaternion kinematics, the RK4 weights) in `jnp.array(value, dtype=...)`
so they take the input array's dtype. The package sets `jax_enable_x64=True`
at import; wrapping prevents inadvertent float32→float64 promotion when
users switch to a float32 backend (e.g., MLX/Metal on Apple Silicon),
which rejects float64 operations.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

from orbitalgym.registry import AttitudeDynamicsKey, register


@flax.struct.dataclass
class AttitudeParams:
    """Per-side rigid-body parameters.

    inertia_diag: principal-axis inertias I_xx, I_yy, I_zz (kg m^2).
    omega_max:    per-axis angular-rate saturation (rad/s).
    """

    inertia_diag: jax.Array  # (3,)
    omega_max: jax.Array  # (3,)


def _omega_to_quat_dot(quat: jax.Array, omega: jax.Array) -> jax.Array:
    """q̇ = ½ Ω(ω) ⊗ q for batched (n,4) quaternion and (n,3) omega.

    Convention: (w, x, y, z), matching state.components.Attitude.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    half = jnp.array(0.5, dtype=quat.dtype)
    qdot_w = half * (-wx * x - wy * y - wz * z)
    qdot_x = half * (wx * w + wz * y - wy * z)
    qdot_y = half * (wy * w - wz * x + wx * z)
    qdot_z = half * (wz * w + wy * x - wx * y)
    return jnp.stack([qdot_w, qdot_x, qdot_y, qdot_z], axis=-1)


def _euler_omega_dot(omega: jax.Array, torque: jax.Array, inertia_diag: jax.Array) -> jax.Array:
    """ω̇ = I⁻¹(τ - ω × (I·ω)) — diagonal-inertia case."""
    i_omega = omega * inertia_diag[None, :]
    gyro = jnp.cross(omega, i_omega)
    return (torque - gyro) / inertia_diag[None, :]


def _rhs(state, torque, params):
    quat, omega = state
    return _omega_to_quat_dot(quat, omega), _euler_omega_dot(omega, torque, params.inertia_diag)


def _rk4_step(state, torque, params, dt):
    dtype = state[0].dtype
    half = jnp.array(0.5, dtype=dtype)
    sixth = jnp.array(1.0 / 6.0, dtype=dtype)
    two = jnp.array(2.0, dtype=dtype)
    dt_ = jnp.array(dt, dtype=dtype)

    k1q, k1o = _rhs(state, torque, params)
    s2 = (state[0] + half * dt_ * k1q, state[1] + half * dt_ * k1o)
    k2q, k2o = _rhs(s2, torque, params)
    s3 = (state[0] + half * dt_ * k2q, state[1] + half * dt_ * k2o)
    k3q, k3o = _rhs(s3, torque, params)
    s4 = (state[0] + dt_ * k3q, state[1] + dt_ * k3o)
    k4q, k4o = _rhs(s4, torque, params)
    quat_new = state[0] + sixth * dt_ * (k1q + two * k2q + two * k3q + k4q)
    omega_new = state[1] + sixth * dt_ * (k1o + two * k2o + two * k3o + k4o)
    return quat_new, omega_new


@register(AttitudeDynamicsKey.RIGID_BODY)
def rigid_body_attitude_step(
    quat: jax.Array,
    omega: jax.Array,
    torque: jax.Array,
    params: AttitudeParams,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Integrate (quat, omega) by dt under applied torque. RK4 + renormalize + clip."""
    quat_new, omega_new = _rk4_step((quat, omega), torque, params, dt)
    quat_new = quat_new / jnp.linalg.norm(quat_new, axis=-1, keepdims=True)
    omega_new = jnp.clip(omega_new, -params.omega_max, params.omega_max)
    return quat_new, omega_new
