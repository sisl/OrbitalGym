"""Ring prior in the RTN frame: in-plane ring, zero cross-track."""

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, Side, make_lady_bandit_guard
from orbitalgym.belief.pf import ParticleFilterRingInitializer
from orbitalgym.reference_orbit import mean_motion


def test_ring_initializer_supports_rtn():
    cfg = make_lady_bandit_guard()
    env = OrbitalGymEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    init = ParticleFilterRingInitializer(
        layout=env.layout,
        ring_radius_m=2000.0,
        mean_motion_rad_s=float(mean_motion(cfg.reference_orbit)),
        n_particles=64,
    )
    belief = init(state, Side.GUARD, jax.random.PRNGKey(1))
    assert belief.particles.shape == (1, 2, 64, 6)
    bandit_cloud = belief.particles[0, 1]  # (64, 6)
    assert jnp.allclose(bandit_cloud[:, 2], 0.0)
    assert jnp.allclose(bandit_cloud[:, 5], 0.0)
    radial_amp = jnp.sqrt(bandit_cloud[:, 0] ** 2 + (bandit_cloud[:, 1] / 2.0) ** 2)
    assert jnp.allclose(radial_amp, 2000.0, rtol=1e-4)
