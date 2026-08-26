import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.dynamics.attitude import AttitudeParams, rigid_body_attitude_step


def _identity_quat(n):
    q = jnp.zeros((n, 4), dtype=jnp.float32)
    return q.at[:, 0].set(jnp.float32(1.0))


def _params(n=1):
    return AttitudeParams(
        inertia_diag=jnp.ones(3, dtype=jnp.float32),
        omega_max=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
    )


def test_zero_torque_zero_omega_invariant():
    quat = _identity_quat(1)
    omega = jnp.zeros((1, 3), dtype=jnp.float32)
    torque = jnp.zeros((1, 3), dtype=jnp.float32)
    quat_new, omega_new = rigid_body_attitude_step(quat, omega, torque, _params(), dt=0.1)
    assert jnp.allclose(quat_new, quat)
    assert jnp.allclose(omega_new, omega)


def test_zero_torque_nonzero_omega_advances_quaternion():
    quat = _identity_quat(1)
    omega = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32)
    torque = jnp.zeros((1, 3), dtype=jnp.float32)
    quat_new, omega_new = rigid_body_attitude_step(quat, omega, torque, _params(), dt=0.1)
    assert jnp.allclose(omega_new, omega, atol=np.float32(1e-6))
    assert quat_new[0, 0] < np.float32(1.0)
    assert quat_new[0, 3] > np.float32(0.0)


def test_quaternion_norm_preserved_over_many_steps():
    quat = _identity_quat(1)
    omega = jnp.array([[0.3, -0.5, 0.7]], dtype=jnp.float32)
    torque = jnp.zeros((1, 3), dtype=jnp.float32)
    p = _params()
    for _ in range(100):
        quat, omega = rigid_body_attitude_step(quat, omega, torque, p, dt=0.05)
    norms = jnp.linalg.norm(quat, axis=-1)
    assert jnp.allclose(norms, np.float32(1.0), atol=np.float32(1e-6))


def test_batch_shape():
    n = 4
    quat = _identity_quat(n)
    omega = jnp.zeros((n, 3), dtype=jnp.float32)
    torque = jnp.zeros((n, 3), dtype=jnp.float32)
    quat_new, omega_new = rigid_body_attitude_step(quat, omega, torque, _params(n), dt=0.1)
    assert quat_new.shape == (n, 4)
    assert omega_new.shape == (n, 3)


def test_jittable():
    f = jax.jit(rigid_body_attitude_step, static_argnames=("dt",))
    quat = _identity_quat(1)
    omega = jnp.zeros((1, 3), dtype=jnp.float32)
    torque = jnp.zeros((1, 3), dtype=jnp.float32)
    f(quat, omega, torque, _params(), 0.1)


def test_constant_torque_saturates_omega():
    """Constant +x torque on a symmetric body ramps ω_x to ω_max, then clips."""
    quat = _identity_quat(1)
    omega = jnp.zeros((1, 3), dtype=jnp.float32)
    torque = jnp.array([[5.0, 0.0, 0.0]], dtype=jnp.float32)
    p = AttitudeParams(
        inertia_diag=jnp.ones(3, dtype=jnp.float32),
        omega_max=jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32),
    )
    for _ in range(2000):
        quat, omega = rigid_body_attitude_step(quat, omega, torque, p, dt=0.01)
    assert jnp.isclose(omega[0, 0], np.float32(0.5), atol=np.float32(1e-5))


def test_free_precession_conserves_angular_momentum_magnitude():
    """Asymmetric body, zero torque, nonzero ω. |L| should be conserved."""
    quat = _identity_quat(1)
    omega = jnp.array([[0.1, 0.2, 0.3]], dtype=jnp.float32)
    torque = jnp.zeros((1, 3), dtype=jnp.float32)
    p = AttitudeParams(
        inertia_diag=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        omega_max=jnp.array([100.0, 100.0, 100.0], dtype=jnp.float32),
    )
    l0_mag = jnp.linalg.norm(omega * p.inertia_diag[None, :])

    for _ in range(1000):
        quat, omega = rigid_body_attitude_step(quat, omega, torque, p, dt=0.01)

    l_mag = jnp.linalg.norm(omega * p.inertia_diag[None, :])
    rel_err = jnp.abs(l_mag - l0_mag) / l0_mag
    assert rel_err < np.float32(1e-4)
