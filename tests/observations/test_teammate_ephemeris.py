"""Tests for TeammateEphemerisObservation."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.belief.pf import ParticleFilterBeliefUpdater, ParticleFilterFromTruthInitializer
from orbitalgym.dynamics.hcw import hcw_rtn_step
from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.observations.composite import CompositeObservation
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.observations.teammate_ephemeris import TeammateEphemerisObservation
from orbitalgym.observations.types import Observation
from orbitalgym.reference_orbit import mean_motion


class _Layout:
    def __init__(self, n_guards, n_bandits, d):
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


class _Env:
    def __init__(self, guards_rtn, bandits_rtn, guards_quat=None):
        self.guards = SimpleNamespace(rtn=guards_rtn, quat=guards_quat)
        self.bandits = SimpleNamespace(rtn=bandits_rtn)


def test_mask_pattern_for_two_guards_one_bandit():
    n_guards, n_bandits, d = 2, 1, 6
    layout = _Layout(n_guards, n_bandits, d)
    env = _Env(
        guards_rtn=jnp.zeros((n_guards, d)),
        bandits_rtn=jnp.zeros((n_bandits, d)),
    )
    fn = TeammateEphemerisObservation(layout=layout, sigma=5.0)
    out = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )
    assert len(out) == 1
    ch = out[0]
    assert isinstance(ch, Observation)
    assert ch.visible.shape == (n_guards, n_guards + n_bandits)
    # visible[i, k] True iff k < n_self (own side) and k != i.
    expected = jnp.array([[False, True, False], [True, False, False]])
    assert jnp.array_equal(ch.visible, expected)


def test_noise_scale_matches_sigma():
    n_guards, n_bandits, d = 2, 1, 6
    layout = _Layout(n_guards, n_bandits, d)
    env = _Env(
        guards_rtn=jnp.zeros((n_guards, d)),
        bandits_rtn=jnp.zeros((n_bandits, d)),
    )
    sigma = 2.5
    fn = TeammateEphemerisObservation(layout=layout, sigma=sigma)
    out = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(1),
        t=jnp.array(0.0),
    )
    ch = out[0]
    assert jnp.allclose(ch.obs_noise, jnp.eye(d) * sigma**2)
    assert jnp.allclose(ch.obs_matrix, jnp.eye(d))
    # obs at the visible teammate pair (0, 1) is truth (=0) plus noise.
    residual = ch.obs[0, 1] - jnp.zeros(d)
    assert bool(jnp.any(jnp.abs(residual) > 1e-6))


def test_composes_inside_composite_with_conical():
    n_guards, n_bandits, d = 2, 1, 6
    layout = _Layout(n_guards, n_bandits, d)
    env = _Env(
        guards_rtn=jnp.zeros((n_guards, d)),
        bandits_rtn=jnp.zeros((n_bandits, d)),
        guards_quat=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (n_guards, 1)),
    )
    teammate = TeammateEphemerisObservation(layout=layout, sigma=5.0)
    cone = ConicalObservation(
        layout=layout,
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]]),
        half_angle_rad=float(jnp.deg2rad(30.0)),
        sigma=1.0,
    )
    comp = CompositeObservation(constituents=(teammate, cone))
    out = comp(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )
    assert len(out) == 2
    assert out[0].obs.shape == (n_guards, n_guards + n_bandits, d)
    assert out[1].obs.shape == (n_guards, n_guards + n_bandits, d)


def test_particle_filter_converges_on_teammate_via_ephemeris_observation():
    """A two-guard scenario: guard 0's PF estimate of guard 1 should
    converge to within 3 sigma of truth after 5 steps of ephemeris updates."""
    cfg = make_lady_bandit_guard(n_guards=2)
    env = OrbitalGymEnv(cfg)
    sigma = 5.0
    obs_fn = TeammateEphemerisObservation(layout=env.layout, sigma=sigma)

    state, _ = env.reset(jax.random.PRNGKey(0))

    n_motion = float(mean_motion(cfg.reference_orbit))
    hcw_params = SimpleNamespace(mean_motion=n_motion)

    def per_vehicle_dyn(x, u, dt):
        return hcw_rtn_step(x[None, :], u[None, :], hcw_params, dt)[0]

    updater = ParticleFilterBeliefUpdater(
        dynamics_fn=per_vehicle_dyn,
        process_noise=jnp.eye(6) * 1e-8,
        dt=cfg.dt,
    )
    init = ParticleFilterFromTruthInitializer(
        layout=env.layout, n_particles=1024, jitter_scale=1e-3
    )
    belief = init(state, Side.GUARD, jax.random.PRNGKey(1))
    # Inject a large *position-only* error into guard 0's estimate of guard 1
    # (index 1 in the n_total axis) so there is real convergence to observe;
    # velocity is left at truth so dynamics propagation does not itself
    # diverge the estimate.
    pos_offset = jax.random.normal(jax.random.PRNGKey(3), (1024, 3)) * 30.0
    particles = belief.particles.at[0, 1, :, :3].add(pos_offset)
    belief = belief.replace(particles=particles)

    zero_guard_cmd = env.guard_command_cls.zeros(cfg.n_guards)
    zero_bandit_cmd = env.bandit_command_cls.zeros(cfg.n_bandits)
    actions = Actions(sides=BySide(guard=zero_guard_cmd, bandit=zero_bandit_cmd))
    zero_dv = jnp.zeros((cfg.n_guards, 3))

    key = jax.random.PRNGKey(2)
    cur_state = state
    for _ in range(5):
        k_step, k_obs, k_upd, key = jax.random.split(key, 4)
        step_out = env.step(k_step, cur_state, actions)
        cur_state = step_out.state
        obs_channels = obs_fn(cur_state, actions, Side.GUARD, env.config, k_obs, cur_state.t)
        belief = updater(belief, obs_channels, zero_dv, Side.GUARD, k_upd)

    est_pos = belief.mean[0, 1, :3]
    true_pos = cur_state.guards.rtn[1, :3]
    err = jnp.linalg.norm(est_pos - true_pos)
    assert float(err) < 3 * sigma
