"""The ring initializer accepts one radius per opposing vehicle."""

import jax
import jax.numpy as jnp

from orbitalgym.belief.pf import ParticleFilterRingInitializer
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Side
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.registry import StateComponentKey
from orbitalgym.sampling.mass import ConstantMass
from orbitalgym.sampling.side import RelativeEllipse
from orbitalgym.sampling.spec import ICSpec


def _env(n_guards: int) -> OrbitalGymEnv:
    cfg = make_lady_bandit_guard(
        n_guards=n_guards,
        n_bandits=1,
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=300.0, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=3000.0, mass_sampler=ConstantMass(propellant_mass_kg=10.0)
            ),
        ),
    )
    return OrbitalGymEnv(cfg)


def test_per_vehicle_radii_place_each_opponent_on_its_own_ring():
    env = _env(2)
    state, _ = env.reset(jax.random.PRNGKey(0))
    init = ParticleFilterRingInitializer(
        layout=env.layout,
        ring_radius_m=jnp.array([300.0, 2000.0]),
        mean_motion_rad_s=env.mean_motion,
        n_particles=64,
    )
    belief = init(state, Side.BANDIT, jax.random.PRNGKey(1))
    opp = belief.particles[0, 1:, :, :3]  # bandit observer, guard targets (2, K, 3)
    radial_amplitude = jnp.max(jnp.abs(opp[..., 0]), axis=-1)
    assert float(radial_amplitude[0]) < 350.0
    assert float(radial_amplitude[1]) > 1500.0


def test_a_scalar_radius_still_broadcasts():
    env = _env(2)
    state, _ = env.reset(jax.random.PRNGKey(0))
    init = ParticleFilterRingInitializer(
        layout=env.layout, ring_radius_m=1000.0, mean_motion_rad_s=env.mean_motion, n_particles=32
    )
    belief = init(state, Side.BANDIT, jax.random.PRNGKey(1))
    assert belief.particles.shape[1] == 3
