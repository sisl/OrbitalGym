"""Tests for belief/pf.py — ParticleFilterBelief + updater + initializers."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.belief.pf import (
    ParticleFilterBelief,
    ParticleFilterBeliefUpdater,
    ParticleFilterFromTruthInitializer,
    ParticleFilterRingInitializer,
)
from orbitalgym.env.types import Side
from orbitalgym.observations.types import Observation
from orbitalgym.reference_orbit import mean_motion

# ---- helpers ------------------------------------------------------------


class _Layout:
    def __init__(self, n_guards: int, n_bandits: int, d: int) -> None:
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


def _identity_dynamics(x, u, dt):
    del u, dt
    return x


def _make_uniform_belief(
    n_obs: int = 1, n_total: int = 2, k: int = 16, d: int = 4
) -> ParticleFilterBelief:
    particles = jnp.zeros((n_obs, n_total, k, d))
    log_w = jnp.full((n_obs, n_total, k), -jnp.log(float(k)))
    n_eff = jnp.full((n_obs, n_total), float(k))
    weight_entropy = jnp.full((n_obs, n_total), float(jnp.log(k)))
    resampled = jnp.zeros((n_obs, n_total), dtype=bool)
    return ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )


# ---- belief mean property ----------------------------------------------


def _trivial_metrics(n_obs: int, n_total: int, k: int):
    n_eff = jnp.full((n_obs, n_total), float(k))
    weight_entropy = jnp.full((n_obs, n_total), float(jnp.log(k)))
    resampled = jnp.zeros((n_obs, n_total), dtype=bool)
    return n_eff, weight_entropy, resampled


def test_mean_property_returns_weighted_average():
    # Two particles at +5 and -5 with equal weight → mean = 0.
    particles = jnp.array([[[[5.0, 0.0, 0.0, 0.0], [-5.0, 0.0, 0.0, 0.0]]]])
    # shape (1, 1, 2, 4)
    log_w = jnp.full((1, 1, 2), -jnp.log(2.0))
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, 2)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )
    assert jnp.allclose(b.mean, jnp.zeros((1, 1, 4)))

    # Re-weight to favour the +5 particle by 100x.
    biased_log_w = jnp.array([[[jnp.log(100.0), 0.0]]])
    biased_log_w = jax.nn.log_softmax(biased_log_w, axis=-1)
    b2 = ParticleFilterBelief(
        particles=particles,
        log_weights=biased_log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )
    # Posterior mean dominated by the +5 particle.
    assert b2.mean[0, 0, 0] > 4.9


# ---- predict-only -------------------------------------------------------


def test_predict_only_with_zero_process_noise_is_identity():
    """With zero Q + identity dynamics, particles should change only by
    the cholesky-regularization noise (1e-6 std)."""
    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((4, 4)),
        dt=1.0,
        n_eff_threshold=0.0,
    )
    b = _make_uniform_belief()
    out = upd(
        b,
        observations=(),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    # Only the chol-regularization (1e-12 I → 1e-6 std) perturbs particles.
    assert jnp.allclose(out.particles, b.particles, atol=1e-4)
    assert jnp.allclose(out.log_weights, b.log_weights)


# ---- weight update on visible measurement -------------------------------


def test_visible_observation_concentrates_weights_near_truth():
    """Two particles, one near the true measurement, one far away. Visible
    update should put nearly all weight on the nearer particle."""
    # 1 obs, 1 target, 2 particles, d=2.
    particles = jnp.array([[[[1.0, 0.0], [-100.0, 0.0]]]])  # (1, 1, 2, 2)
    log_w = jnp.full((1, 1, 2), -jnp.log(2.0))
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, 2)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
    )
    H = jnp.eye(2)  # noqa: N806
    R = jnp.eye(2) * 0.5  # noqa: N806
    chan = Observation(
        obs=jnp.array([[[1.0, 0.0]]]),
        visible=jnp.array([[True]]),
        obs_matrix=H,
        obs_noise=R,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    weights = jax.nn.softmax(out.log_weights, axis=-1)
    assert weights[0, 0, 0] > 0.999  # nearer particle dominates
    assert weights[0, 0, 1] < 1e-3


def test_non_visible_observation_leaves_weights_unchanged():
    """When ``visible=False`` the channel must not alter weights — the
    PF treats non-visible pairs as no-information, not negative info."""
    particles = jnp.array([[[[1.0, 0.0], [-100.0, 0.0]]]])
    log_w = jnp.full((1, 1, 2), -jnp.log(2.0))
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, 2)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
    )
    chan = Observation(
        obs=jnp.array([[[1.0, 0.0]]]),
        visible=jnp.array([[False]]),  # not visible — no update
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2) * 0.5,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    # log_softmax of an already-uniform vector is uniform.
    assert jnp.allclose(out.log_weights, log_w, atol=1e-6)


# ---- nonlinear obs_fn ---------------------------------------------------


def test_nonlinear_obs_fn_uses_obs_fn_for_likelihood():
    """obs_fn = ‖x‖. Two particles, one near radius 1, one near radius 5;
    measurement says r=1. Closer particle should win."""
    particles = jnp.array([[[[1.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]]])
    log_w = jnp.full((1, 1, 2), -jnp.log(2.0))
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, 2)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((4, 4)),
        dt=1.0,
        n_eff_threshold=0.0,
    )

    def range_only(x):
        return jnp.array([jnp.linalg.norm(x[:2])])

    chan = Observation(
        obs=jnp.array([[[1.0]]]),
        visible=jnp.array([[True]]),
        obs_matrix=jnp.zeros((1, 4)),
        obs_noise=jnp.eye(1) * 0.1,
        obs_fn=range_only,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    weights = jax.nn.softmax(out.log_weights, axis=-1)
    assert weights[0, 0, 0] > weights[0, 0, 1]  # particle at r=1 wins


# ---- resampling ---------------------------------------------------------


def test_resample_collapses_to_high_weight_particle_when_n_eff_low():
    """Strongly skewed weights → systematic resampling produces many
    copies of the dominant particle. After resample, weights are uniform."""
    # 1 obs, 1 target, 64 particles in d=2.
    k_particles = 64
    particles = jax.random.normal(jax.random.PRNGKey(1), (1, 1, k_particles, 2)) * 5.0
    # Bias all log-weights toward particle 0.
    log_w_raw = jnp.full((1, 1, k_particles), -50.0).at[0, 0, 0].set(0.0)
    log_w = jax.nn.log_softmax(log_w_raw, axis=-1)
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, k_particles)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.5,  # will trip
    )
    out = upd(
        b,
        observations=(),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(2),
    )
    # All resampled particles should equal the original particle 0.
    target = particles[0, 0, 0]
    matches = jnp.all(jnp.isclose(out.particles[0, 0], target[None, :], atol=1e-6), axis=-1)
    assert int(jnp.sum(matches)) == k_particles
    # Post-resample weights are uniform.
    expected_log_w = -jnp.log(float(k_particles))
    assert jnp.allclose(out.log_weights, expected_log_w)


def test_resample_skipped_when_weights_already_uniform():
    """When N_eff = K, resampling should not fire. Particles are jittered
    only by process noise (zero here), so they stay put."""
    k_particles = 32
    particles = jax.random.normal(jax.random.PRNGKey(3), (1, 1, k_particles, 2)) * 3.0
    log_w = jnp.full((1, 1, k_particles), -jnp.log(float(k_particles)))
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, k_particles)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.5,
    )
    out = upd(
        b,
        observations=(),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(4),
    )
    assert jnp.allclose(out.particles, particles)


# ---- initializers -------------------------------------------------------


def test_pf_from_truth_initializer_anchors_particles_to_truth():
    n_g, n_b, d = 1, 1, 4
    layout = _Layout(n_g, n_b, d)
    truth_g = jnp.array([[10.0, 20.0, 0.1, 0.2]])
    truth_b = jnp.array([[100.0, 200.0, 1.0, 2.0]])
    env_state = SimpleNamespace(
        guards=SimpleNamespace(rt=truth_g),
        bandits=SimpleNamespace(rt=truth_b),
    )
    init = ParticleFilterFromTruthInitializer(layout=layout, n_particles=64, jitter_scale=1e-4)
    belief = init(env_state, side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert belief.particles.shape == (n_g, n_g + n_b, 64, d)
    # Posterior mean should equal truth to within jitter * 1/sqrt(K).
    expected_mean = jnp.concatenate([truth_g, truth_b], axis=0)
    assert jnp.allclose(belief.mean[0], expected_mean, atol=5e-4)


def test_pf_ring_initializer_places_opposing_particles_on_ring():
    """Cross-pair particles should satisfy the natural-motion ring
    relation x^2/a^2 + y^2/(2a)^2 = 1 (within float tolerance)."""
    n_g, n_b, d = 1, 1, 4
    layout = _Layout(n_g, n_b, d)
    # Truth values irrelevant for opposing-side particles, but still required.
    env_state = SimpleNamespace(
        guards=SimpleNamespace(rt=jnp.zeros((n_g, d))),
        bandits=SimpleNamespace(rt=jnp.zeros((n_b, d))),
    )
    a = 2000.0
    n_motion = 1.13e-3  # rough LEO mean motion
    init = ParticleFilterRingInitializer(
        layout=layout,
        ring_radius_m=a,
        mean_motion_rad_s=n_motion,
        n_particles=128,
    )
    belief = init(env_state, side=Side.GUARD, key=jax.random.PRNGKey(0))
    # Opposing-side particles live at indices k >= n_g, here k = 1.
    opp_particles = belief.particles[0, n_g, :, :]  # (128, 4)
    r = opp_particles[:, 0]
    t = opp_particles[:, 1]
    # 2:1 RT ellipse: r = -a cos(phi), t = 2 a sin(phi) → r²/a² + t²/(2a)² = 1.
    residual = (r / a) ** 2 + (t / (2 * a)) ** 2 - 1.0
    assert float(jnp.max(jnp.abs(residual))) < 1e-6
    # Velocity components should match the natural-motion derivative.
    rdot = opp_particles[:, 2]
    tdot = opp_particles[:, 3]
    # rdot = n a sin(phi) and tdot = 2 n a cos(phi)
    # → (rdot / (n a))² + (tdot / (2 n a))² = 1.
    vel_residual = (rdot / (n_motion * a)) ** 2 + (tdot / (2 * n_motion * a)) ** 2 - 1.0
    assert float(jnp.max(jnp.abs(vel_residual))) < 1e-6


def test_pf_ring_initializer_rejects_unsupported_dimensions():
    layout = _Layout(1, 1, 5)
    env_state = SimpleNamespace(
        guards=SimpleNamespace(state_5d=jnp.zeros((1, 5))),
        bandits=SimpleNamespace(state_5d=jnp.zeros((1, 5))),
    )
    init = ParticleFilterRingInitializer(
        layout=layout, ring_radius_m=1.0, mean_motion_rad_s=1.0, n_particles=4
    )
    try:
        init(env_state, side=Side.GUARD, key=jax.random.PRNGKey(0))
    except ValueError as e:
        assert "d in (4, 6)" in str(e)
        return
    raise AssertionError("expected ValueError for d=5")


# ---- collapse + resample tracking metrics -----------------------------


def test_initial_belief_has_max_n_eff_and_no_resample():
    """Fresh belief from any initializer should report N_eff = K,
    weight_entropy = log K, resampled = False everywhere."""
    layout = _Layout(1, 1, 4)
    env_state = SimpleNamespace(
        guards=SimpleNamespace(rt=jnp.zeros((1, 4))),
        bandits=SimpleNamespace(rt=jnp.zeros((1, 4))),
    )
    init_truth = ParticleFilterFromTruthInitializer(layout=layout, n_particles=32)
    b = init_truth(env_state, side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert jnp.allclose(b.n_eff, 32.0)
    assert jnp.allclose(b.weight_entropy, jnp.log(32.0))
    assert not bool(jnp.any(b.resampled))

    init_ring = ParticleFilterRingInitializer(
        layout=layout, ring_radius_m=1000.0, mean_motion_rad_s=1e-3, n_particles=32
    )
    b2 = init_ring(env_state, side=Side.GUARD, key=jax.random.PRNGKey(0))
    assert jnp.allclose(b2.n_eff, 32.0)
    assert jnp.allclose(b2.weight_entropy, jnp.log(32.0))
    assert not bool(jnp.any(b2.resampled))


def test_visible_observation_drops_n_eff_and_records_resample():
    """A sharp likelihood pushes most weight onto one particle. N_eff
    should drop sharply, weight_entropy should drop, and `resampled`
    should be True for the visible pair (since N_eff/K < 0.5)."""
    # 1 obs, 1 target, 16 particles spread over a wide range; truth at 0.
    k_particles = 16
    particles = jnp.linspace(-30.0, 30.0, k_particles).reshape(1, 1, k_particles, 1)
    log_w = jnp.full((1, 1, k_particles), -jnp.log(float(k_particles)))
    n_eff_init, ent_init, res_init = _trivial_metrics(1, 1, k_particles)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff_init,
        weight_entropy=ent_init,
        resampled=res_init,
    )

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((1, 1)),
        dt=1.0,
        n_eff_threshold=0.5,
    )
    chan = Observation(
        obs=jnp.array([[[0.0]]]),
        visible=jnp.array([[True]]),
        obs_matrix=jnp.eye(1),
        obs_noise=jnp.eye(1) * 0.5,  # tight likelihood
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 1)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(7),
    )

    # N_eff dropped from K to a small value (most particles are far from
    # the measurement and get near-zero weight).
    assert float(out.n_eff[0, 0]) < float(k_particles) / 2
    assert float(out.weight_entropy[0, 0]) < float(jnp.log(k_particles))
    # Resample fired for this pair.
    assert bool(out.resampled[0, 0])


def test_no_observations_keeps_n_eff_at_max():
    """Predict-only step with no measurement leaves N_eff = K and does
    not trigger resampling."""
    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((4, 4)),
        dt=1.0,
        n_eff_threshold=0.5,
    )
    b = _make_uniform_belief(k=8)
    out = upd(
        b,
        observations=(),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(9),
    )
    # Predict alone does not change weights → N_eff stays at K.
    assert jnp.allclose(out.n_eff, 8.0, atol=1e-6)
    assert jnp.allclose(out.weight_entropy, jnp.log(8.0), atol=1e-6)
    assert not bool(jnp.any(out.resampled))


# ---- end-to-end with hcw_rt_step ---------------------------------------


def test_pf_predict_with_hcw_rt_keeps_ring_particles_on_ring():
    """Predicting a ring-distributed particle cloud through HCW for one
    short step should leave them on the natural-motion ring (since each
    particle is itself a natural-motion solution)."""
    from orbitalgym.dynamics.hcw import hcw_rt_step

    n_motion = 1.13e-3
    a = 2000.0

    hcw_params = SimpleNamespace(mean_motion=n_motion)

    def per_vehicle_hcw_rt(x, u, dt):
        return hcw_rt_step(x[None, :], u[None, :], hcw_params, dt)[0]

    layout = _Layout(1, 1, 4)
    env_state = SimpleNamespace(
        guards=SimpleNamespace(rt=jnp.zeros((1, 4))),
        bandits=SimpleNamespace(rt=jnp.zeros((1, 4))),
    )
    init = ParticleFilterRingInitializer(
        layout=layout, ring_radius_m=a, mean_motion_rad_s=n_motion, n_particles=64
    )
    b = init(env_state, side=Side.GUARD, key=jax.random.PRNGKey(0))

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=per_vehicle_hcw_rt,
        process_noise=jnp.zeros((4, 4)),
        dt=10.0,
        n_eff_threshold=0.0,
    )
    out = upd(
        b,
        observations=(),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(1),
    )
    # After 10s of HCW propagation, opposing particles should still be on
    # the same ring (natural motion preserves it exactly).
    opp = out.particles[0, 1, :, :]
    r = opp[:, 0]
    t = opp[:, 1]
    residual = (r / a) ** 2 + (t / (2 * a)) ** 2 - 1.0
    assert float(jnp.max(jnp.abs(residual))) < 1e-6


def test_initializers_trace_under_jit_and_vmap():
    """Initializers must build under an outer jit so batched evaluation can compile them."""
    cfg = make_lady_bandit_guard()
    env = OrbitalGymEnv(cfg)
    states, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 3))
    keys = jax.random.split(jax.random.PRNGKey(1), 3)

    def _run_init(init):
        fn = jax.jit(jax.vmap(lambda s, k: init(s, Side.GUARD, k)))
        belief = fn(states, keys)
        assert belief.particles.shape[0] == 3
        assert jnp.allclose(belief.weight_entropy, jnp.log(8.0))

    truth_init = ParticleFilterFromTruthInitializer(layout=env.layout, n_particles=8)
    ring_init = ParticleFilterRingInitializer(
        layout=env.layout,
        ring_radius_m=1000.0,
        mean_motion_rad_s=float(mean_motion(cfg.reference_orbit)),
        n_particles=8,
    )
    _run_init(truth_init)
    _run_init(ring_init)
