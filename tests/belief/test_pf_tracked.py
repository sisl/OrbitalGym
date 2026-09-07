"""Tracked-prior initializer and the ring initializer's along-track offset."""

import jax
import jax.numpy as jnp

from orbitalgym.belief.pf import ParticleFilterRingInitializer, ParticleFilterTrackedInitializer
from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Side
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.registry import StateComponentKey
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec

RING_RADIUS_M = 3000.0
OFFSET_M = 10_000.0


def _env() -> OrbitalGymEnv:
    cfg = make_lady_bandit_guard(
        n_guards=1,
        n_bandits=1,
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=300.0, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=RING_RADIUS_M,
                along_track_offset_m=OFFSET_M,
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
        ),
    )
    return OrbitalGymEnv(cfg)


def test_tracked_prior_cloud_matches_the_requested_sigmas():
    env = _env()
    state, _ = env.reset(jax.random.PRNGKey(0))
    sigma_pos, sigma_vel = 300.0, 0.3
    init = ParticleFilterTrackedInitializer(
        layout=env.layout,
        n_particles=4096,
        sigma_pos_m=sigma_pos,
        sigma_vel_mps=sigma_vel,
    )
    belief = init(state, Side.GUARD, jax.random.PRNGKey(1))

    cloud = belief.particles[0, 1]  # guard observer, bandit target: (K, 6)
    truth = state.bandits.rtn[0]
    k = cloud.shape[0]

    std = jnp.std(cloud, axis=0)
    expected = jnp.array([sigma_pos] * 3 + [sigma_vel] * 3)
    assert jnp.all(jnp.abs(std - expected) / expected < 0.15)

    err = jnp.abs(jnp.mean(cloud, axis=0) - truth)
    assert jnp.all(err < 3.0 * expected / jnp.sqrt(k))

    # Self-pair particles sit at truth up to the tiny jitter.
    own = belief.particles[0, 0]
    assert float(jnp.max(jnp.abs(own - state.guards.rtn[0]))) < 1e-2


def test_tracked_prior_runs_under_jit():
    env = _env()
    state, _ = env.reset(jax.random.PRNGKey(0))
    init = ParticleFilterTrackedInitializer(
        layout=env.layout, n_particles=64, sigma_pos_m=100.0, sigma_vel_mps=0.1
    )
    belief = jax.jit(lambda s, k: init(s, Side.GUARD, k))(state, jax.random.PRNGKey(2))
    assert belief.particles.shape == (1, 2, 64, 6)


def test_ring_offset_shifts_the_cloud_and_keeps_it_on_the_ring():
    env = _env()
    state, _ = env.reset(jax.random.PRNGKey(0))
    init = ParticleFilterRingInitializer(
        layout=env.layout,
        ring_radius_m=RING_RADIUS_M,
        mean_motion_rad_s=env.mean_motion,
        n_particles=2048,
        along_track_offset_m=OFFSET_M,
    )
    belief = jax.jit(lambda s, k: init(s, Side.GUARD, k))(state, jax.random.PRNGKey(3))
    cloud = belief.particles[0, 1]  # (K, 6)

    along_mean = float(jnp.mean(cloud[:, 1]))
    assert abs(along_mean - OFFSET_M) < 2.0 * RING_RADIUS_M / jnp.sqrt(cloud.shape[0]) * 3.0
    assert float(jnp.max(jnp.abs(cloud[:, 1] - OFFSET_M))) < 2.1 * RING_RADIUS_M

    # A free-drift step keeps the cloud on the same ring, still centred on
    # the offset: the radial amplitude and the along-track mean are invariant.
    stepped = cloud @ hcw_rtn_stm(env.mean_motion, 60.0).T

    before = float(jnp.max(jnp.abs(cloud[:, 0])))
    after = float(jnp.max(jnp.abs(stepped[:, 0])))
    assert abs(after - before) / before < 0.05
    assert abs(float(jnp.mean(stepped[:, 1])) - along_mean) < 0.02 * RING_RADIUS_M
