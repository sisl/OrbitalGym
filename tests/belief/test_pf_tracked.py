"""Tracked-prior initializer and the ring initializer's along-track offset."""

import jax
import jax.numpy as jnp

from orbitalgym.belief.pf import ParticleFilterRingInitializer, ParticleFilterTrackedInitializer
from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Side
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.registry import DynamicsKey, Frame, StateComponentKey
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


def _rt_env() -> OrbitalGymEnv:
    """1v1 in the 4-state RT plane: dynamics_state_dim is 4."""
    cfg = make_lady_bandit_guard(
        n_guards=1,
        n_bandits=1,
        truth_dynamics=DynamicsKey.HCW_RT,
        policy_dynamics=DynamicsKey.HCW_RT,
        action_frame=Frame.RT,
        guard_components=(StateComponentKey.RT, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RT,),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=300.0, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=RING_RADIUS_M, along_track_offset_m=OFFSET_M
            ),
        ),
    )
    return OrbitalGymEnv(cfg)


def _two_bandit_env() -> OrbitalGymEnv:
    cfg = make_lady_bandit_guard(
        n_guards=1,
        n_bandits=2,
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=300.0, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=RING_RADIUS_M, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
            ),
        ),
    )
    return OrbitalGymEnv(cfg)


def test_tracked_sigmas_split_position_and_velocity_in_the_rt_plane():
    env = _rt_env()
    assert env.layout.dynamics_state_dim == 4
    state, _ = env.reset(jax.random.PRNGKey(0))
    sigma_pos, sigma_vel = 500.0, 0.5
    init = ParticleFilterTrackedInitializer(
        layout=env.layout, n_particles=4096, sigma_pos_m=sigma_pos, sigma_vel_mps=sigma_vel
    )
    belief = init(state, Side.GUARD, jax.random.PRNGKey(1))

    cloud = belief.particles[0, 1]  # (K, 4) — [r, t, r_dot, t_dot]
    expected = jnp.array([sigma_pos, sigma_pos, sigma_vel, sigma_vel])
    std = jnp.std(cloud, axis=0)
    assert jnp.all(jnp.abs(std - expected) / expected < 0.15)

    err = jnp.abs(jnp.mean(cloud, axis=0) - state.bandits.rt[0])
    assert jnp.all(err < 3.0 * expected / jnp.sqrt(cloud.shape[0]))

    own = belief.particles[0, 0]
    assert float(jnp.max(jnp.abs(own - state.guards.rt[0]))) < 1e-2


def test_per_opponent_radii_and_offsets_place_each_cloud_on_its_own_ring():
    env = _two_bandit_env()
    state, _ = env.reset(jax.random.PRNGKey(0))
    radii = jnp.array([500.0, 4000.0])
    offsets = jnp.array([0.0, OFFSET_M])
    init = ParticleFilterRingInitializer(
        layout=env.layout,
        ring_radius_m=radii,
        mean_motion_rad_s=env.mean_motion,
        n_particles=2048,
        along_track_offset_m=offsets,
    )
    belief = init(state, Side.GUARD, jax.random.PRNGKey(1))
    clouds = belief.particles[0, 1:]  # (2, K, 6)

    for i in range(2):
        radius = float(radii[i])
        offset = float(offsets[i])
        radial_amplitude = float(jnp.max(jnp.abs(clouds[i, :, 0])))
        assert abs(radial_amplitude - radius) / radius < 0.05
        centred = clouds[i, :, 1] - offset
        assert float(jnp.max(jnp.abs(centred))) < 2.05 * radius
        assert abs(float(jnp.mean(clouds[i, :, 1])) - offset) < 0.2 * radius


def test_every_ring_particle_satisfies_the_natural_motion_relations():
    env = _env()
    state, _ = env.reset(jax.random.PRNGKey(0))
    n_motion = env.mean_motion
    init = ParticleFilterRingInitializer(
        layout=env.layout,
        ring_radius_m=RING_RADIUS_M,
        mean_motion_rad_s=n_motion,
        n_particles=512,
        along_track_offset_m=OFFSET_M,
    )
    belief = init(state, Side.GUARD, jax.random.PRNGKey(4))
    cloud = belief.particles[0, 1]

    # With t' = t - offset: r = -a cos p, t' = 2 a sin p, r_dot = n a sin p,
    # t_dot = 2 n a cos p, so t_dot = -2 n r and r_dot = n t' / 2, per particle.
    r = cloud[:, 0]
    t_centred = cloud[:, 1] - OFFSET_M
    tol = 1e-3 * n_motion * RING_RADIUS_M
    assert float(jnp.max(jnp.abs(cloud[:, 4] + 2.0 * n_motion * r))) < tol
    assert float(jnp.max(jnp.abs(cloud[:, 3] - 0.5 * n_motion * t_centred))) < tol
    # Cross-track stays exactly zero, and each particle sits on the ring.
    assert float(jnp.max(jnp.abs(cloud[:, 2]))) == 0.0
    assert float(jnp.max(jnp.abs(cloud[:, 5]))) == 0.0
    amplitude = jnp.sqrt(r**2 + (0.5 * t_centred) ** 2)
    assert float(jnp.max(jnp.abs(amplitude - RING_RADIUS_M))) < 1e-2 * RING_RADIUS_M


def test_tracked_prior_is_reproducible_and_key_dependent():
    env = _env()
    state, _ = env.reset(jax.random.PRNGKey(0))
    init = ParticleFilterTrackedInitializer(
        layout=env.layout, n_particles=128, sigma_pos_m=300.0, sigma_vel_mps=0.3
    )
    a = init(state, Side.GUARD, jax.random.PRNGKey(7))
    a_again = init(state, Side.GUARD, jax.random.PRNGKey(7))
    b = init(state, Side.GUARD, jax.random.PRNGKey(8))

    assert jnp.array_equal(a.particles, a_again.particles)
    assert not jnp.array_equal(a.particles, b.particles)
    # The self-pair jitter and the cross-pair track noise come from separate
    # splits, so both blocks move when the key changes.
    assert not jnp.array_equal(a.particles[0, 0], b.particles[0, 0])
    assert not jnp.array_equal(a.particles[0, 1], b.particles[0, 1])
