import jax
import jax.numpy as jnp

from orbitalgym.sampling.attitude import (
    FixedAttitude,
    IdentityAttitude,
    UniformAttitude,
    UniformAttitudeAndRates,
    UniformBodyRates,
)


def test_identity_returns_identity_quat_and_zero_omega():
    sampler = IdentityAttitude()
    quat, omega = sampler(n_vehicles=3, key=jax.random.PRNGKey(0))
    assert quat.shape == (3, 4)
    assert omega.shape == (3, 3)
    assert jnp.allclose(quat, jnp.array([[1.0, 0.0, 0.0, 0.0]] * 3))
    assert jnp.allclose(omega, 0.0)


def test_fixed_broadcasts_quat_and_omega():
    sampler = FixedAttitude(
        quat_wxyz=(0.5, 0.5, 0.5, 0.5),  # 120° about (1,1,1)/sqrt(3)
        omega_rad_s=(0.0, 0.0, float(jnp.deg2rad(0.1))),
    )
    quat, omega = sampler(n_vehicles=2, key=jax.random.PRNGKey(0))
    assert quat.shape == (2, 4)
    assert jnp.allclose(jnp.linalg.norm(quat, axis=-1), 1.0)
    assert jnp.allclose(omega[:, 2], jnp.deg2rad(0.1))


def test_uniform_quat_unit_norm_and_w_nonnegative():
    sampler = UniformAttitude()
    quat, omega = sampler(n_vehicles=64, key=jax.random.PRNGKey(0))
    norms = jnp.linalg.norm(quat, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)
    # Canonical: w >= 0 for every vehicle.
    assert bool(jnp.all(quat[:, 0] >= 0.0))
    assert jnp.allclose(omega, 0.0)


def test_uniform_rates_within_box():
    omega_max = (0.1, 0.2, 0.3)
    sampler = UniformBodyRates(omega_max_rad_s=omega_max)
    quat, omega = sampler(n_vehicles=128, key=jax.random.PRNGKey(0))
    bound = jnp.asarray(omega_max)
    assert bool(jnp.all(jnp.abs(omega) <= bound))
    assert jnp.allclose(quat, jnp.array([[1.0, 0.0, 0.0, 0.0]] * 128))


def test_uniform_quat_and_rates_combined():
    sampler = UniformAttitudeAndRates(omega_max_rad_s=(0.5, 0.5, 0.5))
    quat, omega = sampler(n_vehicles=32, key=jax.random.PRNGKey(0))
    norms = jnp.linalg.norm(quat, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)
    assert bool(jnp.all(jnp.abs(omega) <= 0.5))


def test_pytrees_jittable():
    """Each sampler must be jit-friendly."""
    f = jax.jit(lambda key: IdentityAttitude()(n_vehicles=4, key=key))
    f(jax.random.PRNGKey(0))
    f2 = jax.jit(lambda key: UniformAttitude()(n_vehicles=4, key=key))
    f2(jax.random.PRNGKey(0))
