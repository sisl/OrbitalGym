import jax.numpy as jnp

from orbital_game.dynamics.attitude import AttitudeParams


def test_attitude_params_construction():
    p = AttitudeParams(
        inertia_diag=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        omega_max=jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32),
    )
    assert p.inertia_diag.shape == (3,)
    assert p.omega_max.shape == (3,)


def test_attitude_params_is_pytree():
    """flax.struct.dataclass means AttitudeParams threads through jit/vmap."""
    import jax

    p = AttitudeParams(
        inertia_diag=jnp.ones(3, dtype=jnp.float32),
        omega_max=jnp.ones(3, dtype=jnp.float32),
    )
    leaves = jax.tree_util.tree_leaves(p)
    assert len(leaves) == 2
