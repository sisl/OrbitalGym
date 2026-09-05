"""Tests for PF negative-information updates (Off / Hard / Soft modes)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbitalgym.belief.pf import (
    ParticleFilterBelief,
    ParticleFilterBeliefUpdater,
)
from orbitalgym.env.types import Side
from orbitalgym.observations.negative_info import Hard, Off, Soft
from orbitalgym.observations.types import Observation


def _identity_dynamics(x, u, dt):
    del u, dt
    return x


def _trivial_metrics(n_obs: int, n_total: int, k: int):
    n_eff = jnp.full((n_obs, n_total), float(k))
    weight_entropy = jnp.full((n_obs, n_total), float(jnp.log(k)))
    resampled = jnp.zeros((n_obs, n_total), dtype=bool)
    return n_eff, weight_entropy, resampled


def _make_belief(particles: jax.Array) -> ParticleFilterBelief:
    n_obs, n_total, k, _ = particles.shape
    log_w = jnp.full((n_obs, n_total, k), -jnp.log(float(k)))
    n_eff, weight_entropy, resampled = _trivial_metrics(n_obs, n_total, k)
    return ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )


def test_default_negative_info_is_off():
    """The PF must default to Off so existing notebooks/tests are unaffected."""
    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
    )
    assert isinstance(upd.negative_info, Off)


def test_off_mode_with_non_visible_gated_channel_leaves_weights_unchanged():
    """With Off mode, a non-visible gated channel must NOT update weights —
    bit-identical to today's behavior."""
    particles = jnp.array([[[[1.0, 0.0], [-100.0, 0.0]]]])  # (1, 1, 2, 2)
    b = _make_belief(particles)

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Off(),
    )

    # Channel WITH visibility_score_fn but Off mode → score should be ignored.
    def score_fn(particles):
        # Particle 0 is "inside" (score > 0), particle 1 is "outside" (score < 0).
        return jnp.array([[[10.0, -10.0]]])

    chan = Observation(
        obs=jnp.array([[[1.0, 0.0]]]),
        visible=jnp.array([[False]]),  # non-visible
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2) * 0.5,
        visibility_score_fn=score_fn,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    # Off mode: weights unchanged on non-visible pair, even though score_fn is set.
    assert jnp.allclose(out.log_weights, b.log_weights, atol=1e-6)


def test_hard_mode_crushes_inside_particles_on_non_detection():
    """Hard mode: particle whose score > 0 (inside the gate) must be
    crushed when the channel is non-visible."""
    particles = jnp.array([[[[1.0, 0.0], [-1.0, 0.0]]]])  # (1, 1, 2, 2)
    b = _make_belief(particles)

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Hard(),
    )

    def score_fn(parts):
        return jnp.array([[[1000.0, -1000.0]]])

    chan = Observation(
        obs=jnp.array([[[0.0, 0.0]]]),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
        visibility_score_fn=score_fn,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    weights = jax.nn.softmax(out.log_weights, axis=-1)
    # Inside particle (idx 0) crushed; outside particle (idx 1) keeps the mass.
    assert weights[0, 0, 0] < 1e-6
    assert weights[0, 0, 1] > 0.999


def test_hard_mode_boundary_particle_not_crushed():
    """Particle exactly at the boundary (score = 0) must NOT be crushed —
    the comparison is strict (`score > 0`)."""
    particles = jnp.array([[[[1.0, 0.0], [-1.0, 0.0]]]])
    b = _make_belief(particles)

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Hard(),
    )

    def score_fn(parts):
        return jnp.array([[[0.0, -1000.0]]])  # particle 0 exactly on boundary

    chan = Observation(
        obs=jnp.array([[[0.0, 0.0]]]),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
        visibility_score_fn=score_fn,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    # Both particles untouched (boundary is NOT crushed); after log_softmax,
    # the original uniform weights are preserved.
    assert jnp.allclose(out.log_weights, b.log_weights, atol=1e-6)


def test_hard_mode_with_visible_pair_uses_standard_likelihood():
    """When visible=True, Hard mode must NOT trigger negative-info — the
    standard Gaussian likelihood update fires instead."""
    particles = jnp.array([[[[1.0, 0.0], [-100.0, 0.0]]]])
    b = _make_belief(particles)

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Hard(),
    )

    def score_fn(parts):
        return jnp.array([[[1000.0, -1000.0]]])

    chan = Observation(
        obs=jnp.array([[[1.0, 0.0]]]),
        visible=jnp.array([[True]]),  # visible — detection
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2) * 0.5,
        visibility_score_fn=score_fn,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    weights = jax.nn.softmax(out.log_weights, axis=-1)
    # Particle near the measurement (idx 0) wins via standard likelihood.
    # Its score of +1000 would have crushed it had negative info fired.
    assert weights[0, 0, 0] > 0.9


def test_soft_mode_monotone_in_score():
    """Soft mode: log-weight contribution should be monotone-decreasing in
    score (higher score = closer to inside gate = more crushing)."""
    softness = 100.0
    scores = jnp.array([[[-200.0, -100.0, 0.0, 100.0, 200.0]]])
    n_particles = 5

    particles = jnp.zeros((1, 1, n_particles, 2))
    log_w_init = jnp.full((1, 1, n_particles), -jnp.log(float(n_particles)))
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, n_particles)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w_init,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Soft(softness_per_channel=(softness,)),
    )

    def score_fn(parts):
        return scores

    chan = Observation(
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
        visibility_score_fn=score_fn,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    weights = jax.nn.softmax(out.log_weights, axis=-1)[0, 0]  # (5,)
    diffs = jnp.diff(weights)
    assert jnp.all(diffs < 0), f"weights not monotone-decreasing: {weights}"


def test_soft_mode_larger_softness_flattens_weights():
    """Larger softness → flatter post-softmax weight distribution at fixed scores.
    Tight softness (=10) creates a sharp sigmoid transition near score=0, so the
    post-softmax weight range across [-100, 0, 100] is large. Loose softness (=1000)
    flattens the sigmoid, shrinking the weight range."""
    scores = jnp.array([[[-100.0, 0.0, 100.0]]])
    particles = jnp.zeros((1, 1, 3, 2))
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, 3)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=jnp.full((1, 1, 3), -jnp.log(3.0)),
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )

    def score_fn(parts):
        return scores

    chan = Observation(
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
        visibility_score_fn=score_fn,
    )

    def run_with_softness(s: float):
        upd = ParticleFilterBeliefUpdater(
            dynamics_fn=_identity_dynamics,
            process_noise=jnp.zeros((2, 2)),
            dt=1.0,
            n_eff_threshold=0.0,
            negative_info=Soft(softness_per_channel=(s,)),
        )
        out = upd(
            b,
            observations=(chan,),
            action=jnp.zeros((1, 2)),
            side=Side.GUARD,
            key=jax.random.PRNGKey(0),
        )
        return jax.nn.softmax(out.log_weights, axis=-1)[0, 0]

    weights_tight = run_with_softness(10.0)
    weights_loose = run_with_softness(1000.0)
    range_tight = float(weights_tight[0] - weights_tight[2])
    range_loose = float(weights_loose[0] - weights_loose[2])
    assert range_tight > range_loose, (
        f"tight softness should produce larger weight range than loose; "
        f"got tight={range_tight}, loose={range_loose}"
    )


def test_soft_mode_no_nan_on_saturated_cloud():
    """All particles deep inside gate → all crushed equally → softmax must
    still produce finite, valid weights (post-softmax, they're uniform)."""
    scores = jnp.array([[[1e6, 1e6, 1e6]]])
    particles = jnp.zeros((1, 1, 3, 2))
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, 3)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=jnp.full((1, 1, 3), -jnp.log(3.0)),
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Soft(softness_per_channel=(50.0,)),
    )

    def score_fn(parts):
        return scores

    chan = Observation(
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
        visibility_score_fn=score_fn,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    assert jnp.all(jnp.isfinite(out.log_weights))
    weights = jax.nn.softmax(out.log_weights, axis=-1)
    assert jnp.allclose(jnp.sum(weights, axis=-1), 1.0, atol=1e-6)


def test_hard_mode_no_nan_on_saturated_cloud():
    """All particles deep inside gate → Hard crushes all → softmax stays valid."""
    scores = jnp.array([[[1e6, 1e6, 1e6]]])
    particles = jnp.zeros((1, 1, 3, 2))
    n_eff, weight_entropy, resampled = _trivial_metrics(1, 1, 3)
    b = ParticleFilterBelief(
        particles=particles,
        log_weights=jnp.full((1, 1, 3), -jnp.log(3.0)),
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Hard(),
    )

    def score_fn(parts):
        return scores

    chan = Observation(
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
        visibility_score_fn=score_fn,
    )
    out = upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    assert jnp.all(jnp.isfinite(out.log_weights))


def test_pf_rejects_hard_with_no_gated_channels():
    """Mode != Off but no channel supplies visibility_score_fn → ValueError."""
    particles = jnp.zeros((1, 1, 2, 2))
    b = _make_belief(particles)

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Hard(),
    )
    chan = Observation(
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
    )
    with pytest.raises(ValueError, match="no observation channel"):
        upd(
            b,
            observations=(chan,),
            action=jnp.zeros((1, 2)),
            side=Side.GUARD,
            key=jax.random.PRNGKey(0),
        )


def test_pf_rejects_soft_softness_length_mismatch():
    """Soft.softness_per_channel length must equal len(observations)."""
    particles = jnp.zeros((1, 1, 2, 2))
    b = _make_belief(particles)

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Soft(softness_per_channel=(100.0, 200.0)),  # 2 entries
    )

    def score_fn(parts):
        return jnp.zeros((1, 1, 2))

    chan = Observation(  # only 1 channel
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
        visibility_score_fn=score_fn,
    )
    with pytest.raises(ValueError, match="softness_per_channel"):
        upd(
            b,
            observations=(chan,),
            action=jnp.zeros((1, 2)),
            side=Side.GUARD,
            key=jax.random.PRNGKey(0),
        )


def test_pf_off_mode_does_not_require_gated_channels():
    """Off mode must not raise even if no channel is gated — that's the
    whole point of being able to opt out."""
    particles = jnp.zeros((1, 1, 2, 2))
    b = _make_belief(particles)

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Off(),
    )
    chan = Observation(
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
    )
    # Should not raise.
    upd(
        b,
        observations=(chan,),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )


def test_multichannel_mixed_gating_handles_none_score_fn():
    """A PF with two channels — one gated (RangeLimitedObservation-like),
    one non-gated (GPS-like) — must process both correctly. The non-gated
    channel's None visibility_score_fn must not crash the negative-info path."""
    particles = jnp.array([[[[1.0, 0.0], [-1.0, 0.0]]]])  # (1, 1, 2, 2)
    b = _make_belief(particles)

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((2, 2)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Soft(softness_per_channel=(100.0, 0.0001)),  # second is dummy
    )

    def score_fn(parts):
        return jnp.array([[[1000.0, -1000.0]]])

    gated_chan = Observation(
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[False]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
        visibility_score_fn=score_fn,
    )
    # Non-gated channel — visible=True everywhere (GPS-like), no score_fn.
    nongated_chan = Observation(
        obs=jnp.array([[[0.0, 0.0]]]),
        visible=jnp.array([[True]]),  # always visible
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2) * 1.0,
        # visibility_score_fn defaults to None
    )

    out = upd(
        b,
        observations=(gated_chan, nongated_chan),
        action=jnp.zeros((1, 2)),
        side=Side.GUARD,
        key=jax.random.PRNGKey(0),
    )
    # Must not raise. Output weights are finite and normalized.
    assert jnp.all(jnp.isfinite(out.log_weights))
    weights = jax.nn.softmax(out.log_weights, axis=-1)
    assert jnp.allclose(jnp.sum(weights, axis=-1), 1.0, atol=1e-6)
