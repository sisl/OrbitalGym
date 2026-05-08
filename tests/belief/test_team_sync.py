import jax.numpy as jnp

from orbital_game.belief.ekf import EKFBelief
from orbital_game.belief.kf import KFBelief
from orbital_game.belief.pf import ParticleFilterBelief
from orbital_game.belief.sync import (
    BeliefSyncFn,
    EKFTeamFusion,
    KFTeamFusion,
    PFTeamFusion,
)
from orbital_game.registry import BeliefSyncKey, resolve


class _IdentitySync:
    def __call__(self, belief, contact):
        return belief


def test_protocol_runtime_check():
    sync: BeliefSyncFn = _IdentitySync()
    out = sync(belief=object(), contact=jnp.zeros((2,), dtype=jnp.bool_))
    assert out is not None


def test_kf_team_fusion_two_observers_in_contact_tightens():
    # Two observers, one opposing target. Each observer has same mean,
    # different covariances. The pooled posterior should be tighter than
    # either input.
    n_obs, n_total, d = 2, 3, 6  # 2 self + 1 opp
    mean = jnp.zeros((n_obs, n_total, d))
    cov_a = jnp.eye(d) * 4.0
    cov_b = jnp.eye(d) * 1.0
    cov = jnp.stack(
        [
            jnp.stack([cov_a, cov_a, cov_a]),  # observer 0's covs
            jnp.stack([cov_b, cov_b, cov_b]),  # observer 1's covs
        ]
    )
    belief = KFBelief(mean=mean, cov=cov)
    contact = jnp.asarray([True, True])

    fusion = KFTeamFusion()
    fused = fusion(belief, contact)

    # Each in-contact observer's posterior over the OPP target (k=2)
    # should have trace strictly less than min(input traces).
    traces = jnp.trace(fused.cov[:, 2, :, :], axis1=-1, axis2=-2)
    assert float(traces[0]) < float(jnp.trace(cov_a))
    assert float(traces[1]) < float(jnp.trace(cov_a))


def test_kf_team_fusion_solo_observer_is_identity():
    n_obs, n_total, d = 2, 3, 6
    mean = jnp.ones((n_obs, n_total, d))
    cov = jnp.broadcast_to(jnp.eye(d) * 2.0, (n_obs, n_total, d, d))
    belief = KFBelief(mean=mean, cov=cov)
    contact = jnp.asarray([True, False])

    fusion = KFTeamFusion()
    fused = fusion(belief, contact)

    assert jnp.allclose(fused.mean, mean)
    assert jnp.allclose(fused.cov, cov)


def test_kf_team_fusion_no_observers_in_contact_is_identity():
    n_obs, n_total, d = 2, 3, 6
    mean = jnp.ones((n_obs, n_total, d))
    cov = jnp.broadcast_to(jnp.eye(d) * 2.0, (n_obs, n_total, d, d))
    belief = KFBelief(mean=mean, cov=cov)
    contact = jnp.asarray([False, False])

    fusion = KFTeamFusion()
    fused = fusion(belief, contact)

    assert jnp.allclose(fused.mean, mean)
    assert jnp.allclose(fused.cov, cov)


def test_ekf_team_fusion_two_observers_in_contact_tightens():
    n_obs, n_total, d = 2, 3, 6
    mean = jnp.zeros((n_obs, n_total, d))
    cov_a = jnp.eye(d) * 4.0
    cov_b = jnp.eye(d) * 1.0
    cov = jnp.stack(
        [
            jnp.stack([cov_a, cov_a, cov_a]),
            jnp.stack([cov_b, cov_b, cov_b]),
        ]
    )
    belief = EKFBelief(mean=mean, cov=cov)
    contact = jnp.asarray([True, True])

    fusion = EKFTeamFusion()
    fused = fusion(belief, contact)

    traces = jnp.trace(fused.cov[:, 2, :, :], axis1=-1, axis2=-2)
    assert float(traces[0]) < float(jnp.trace(cov_a))
    assert float(traces[1]) < float(jnp.trace(cov_a))


def test_pf_team_fusion_two_observers_pool_particles():
    n_obs, n_total, K, d = 2, 3, 128, 6  # noqa: N806 - K is particle-count math convention
    # Observer 0: particles clustered at +1; observer 1: particles at -1.
    particles_0 = jnp.ones((n_total, K, d))
    particles_1 = -jnp.ones((n_total, K, d))
    particles = jnp.stack([particles_0, particles_1])  # (2, 3, K, d)
    log_w = jnp.full((n_obs, n_total, K), -jnp.log(float(K)))  # uniform
    n_eff = jnp.full((n_obs, n_total), float(K))
    weight_entropy = jnp.full((n_obs, n_total), float(jnp.log(K)))
    resampled = jnp.zeros((n_obs, n_total), dtype=bool)
    belief = ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )
    contact = jnp.asarray([True, True])

    fusion = PFTeamFusion()
    fused = fusion(belief, contact)

    # After pooling + joint resample, each in-contact observer's particles
    # should span both clusters (mean position roughly 0).
    fused_mean_0 = fused.particles[0, 2].mean(axis=0)  # mean over K for target k=2
    assert abs(float(fused_mean_0[0])) < 0.3


def test_kf_team_fusion_resolves_through_registry():
    cls = resolve(BeliefSyncKey.KF_TEAM_FUSION)
    assert cls is KFTeamFusion


def test_ekf_team_fusion_resolves_through_registry():
    cls = resolve(BeliefSyncKey.EKF_TEAM_FUSION)
    assert cls is EKFTeamFusion


def test_pf_team_fusion_resolves_through_registry():
    cls = resolve(BeliefSyncKey.PF_TEAM_FUSION)
    assert cls is PFTeamFusion
