"""BeliefSyncFn protocol and concrete team-fusion implementations.

A `BeliefSyncFn` runs once per side per tick inside `belief_rollout`,
after the per-side belief update and before policy invocation. It is
responsible for pooling teammates' beliefs about the opposing side
when they are simultaneously in contact.

Implementations (added in later tasks):

  - `KFTeamFusion` — information-form pooling for KFBelief.
  - `EKFTeamFusion` — same algorithm, EKFBelief shapes.
  - `PFTeamFusion` — joint resample for ParticleFilterBelief.

See `superpowers/specs/2026-05-07-ground-station-comms-design.md` for
the operational rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from orbital_game.registry import BeliefSyncKey, register


class BeliefSyncFn(Protocol):
    """Pool teammate beliefs about the opposing side during contact.

    `belief` is the side's existing belief pytree. `contact` is a
    per-teammate bool mask: `contact[i]` is True iff teammate i is in a
    contact window this tick. Returns a same-shape belief.
    """

    def __call__(self, belief: Any, contact: jax.Array) -> Any: ...


@register(BeliefSyncKey.KF_TEAM_FUSION)
@dataclass(frozen=True)
class KFTeamFusion:
    """Information-form pooling for KFBelief / EKFBelief.

    For each opposing target k, observers in contact share information
    state (P^-1 mu, P^-1); the pooled posterior overwrites every in-contact
    observer's (mu_i, P_i) for that k. Out-of-contact observers untouched.

    Optimal under independent observations. For known-correlated cases
    (e.g. all teammates pulling from the same operator estimate), use
    covariance intersection instead (added under a separate registry key).

    Same-pair (i == k) observer-on-self entries are excluded from fusion
    -- those slots reflect own-truth and shouldn't be pooled.
    """

    def __call__(self, belief: Any, contact: jax.Array) -> Any:
        # Defensive: skip the work entirely when 0 or 1 observer in contact.
        # `lax.cond` keeps jit happy.
        return jax.lax.cond(
            jnp.sum(contact.astype(jnp.int32)) > 1,
            lambda: self._fuse(belief, contact),
            lambda: belief,
        )

    def _fuse(self, belief: Any, contact: jax.Array) -> Any:
        mean = belief.mean  # (N_obs, N_total, d)
        cov = belief.cov  # (N_obs, N_total, d, d)
        n_obs, n_total, d = mean.shape

        # info[i, k] = P_ik^-1; info_state[i, k] = P_ik^-1 mu_ik
        info = jnp.linalg.inv(cov)
        info_state = jnp.einsum("ikab,ikb->ika", info, mean)

        # Opposing-target mask: targets at indices [n_obs, n_total) are opponents.
        # Teammate-on-teammate cells (k < n_obs) are NOT fused — they reflect
        # truth-tracking on own side, not opponent state.
        k_idx = jnp.arange(n_total)
        opposing = k_idx >= n_obs  # (n_total,)
        contributing = contact[:, None] & opposing[None, :]  # (n_obs, n_total)

        # For each target k, sum over observer axis.
        weight = contributing.astype(info.dtype)
        info_pooled_k = jnp.einsum("ik,ikab->kab", weight, info)
        info_state_pooled_k = jnp.einsum("ik,ika->ka", weight, info_state)

        # Add a per-target eps * I floor so a single-observer-target cell
        # (no contributors) doesn't blow up; we'll only USE the pooled
        # values on rows where contact[i] is True.
        eps_floor = jnp.eye(d) * 1e-8
        info_pooled_k = info_pooled_k + eps_floor

        cov_pooled_k = jnp.linalg.inv(info_pooled_k)  # (N_total, d, d)
        mean_pooled_k = jnp.einsum("kab,kb->ka", cov_pooled_k, info_state_pooled_k)

        # Broadcast pooled-per-k back over the observer axis, then
        # `where` against the original on the contact mask.
        cov_pooled = jnp.broadcast_to(cov_pooled_k[None, ...], cov.shape)
        mean_pooled = jnp.broadcast_to(mean_pooled_k[None, ...], mean.shape)

        # Apply only on (in-contact observer, opposing target) cells.
        apply_mask_2d = contributing
        apply_mask_mean = apply_mask_2d[..., None]
        apply_mask_cov = apply_mask_2d[..., None, None]

        new_mean = jnp.where(apply_mask_mean, mean_pooled, mean)
        new_cov = jnp.where(apply_mask_cov, cov_pooled, cov)

        return belief.replace(mean=new_mean, cov=new_cov)


@register(BeliefSyncKey.EKF_TEAM_FUSION)
@dataclass(frozen=True)
class EKFTeamFusion(KFTeamFusion):
    """Information-form pooling for EKFBelief -- same algorithm as KFTeamFusion.

    Distinct class so the registry has a stable name for serialization. The
    fusion math is identical: KFBelief and EKFBelief both expose the
    (N_obs, N_total, d) mean and (N_obs, N_total, d, d) cov layout.
    """


@register(BeliefSyncKey.PF_TEAM_FUSION)
@dataclass(frozen=True)
class PFTeamFusion:
    """Joint-resample fusion for `ParticleFilterBelief`.

    For each opposing target k, gather particles from every in-contact
    observer (excluding self-pairs), build a combined cloud weighted by
    each observer's normalized weights, and systematic-resample to K
    particles per in-contact observer.

    Out-of-contact observers untouched.
    """

    def __call__(self, belief: Any, contact: jax.Array) -> Any:
        return jax.lax.cond(
            jnp.sum(contact.astype(jnp.int32)) > 1,
            lambda: self._fuse(belief, contact),
            lambda: belief,
        )

    def _fuse(self, belief: Any, contact: jax.Array) -> Any:
        particles = belief.particles  # (N_obs, N_total, K, d)
        log_w = belief.log_weights  # (N_obs, N_total, K)
        n_obs, n_total, K, d = particles.shape  # noqa: N806 - K is particle-count math convention

        # Opposing-target mask: targets at indices [n_obs, n_total) are opponents.
        # Teammate-on-teammate cells (k < n_obs) are NOT fused — they reflect
        # truth-tracking on own side, not opponent state.
        k_idx = jnp.arange(n_total)
        opposing = k_idx >= n_obs  # (n_total,)
        # contributing[i, k]: include observer i's contribution for target k
        contributing = contact[:, None] & opposing[None, :]  # (n_obs, n_total)

        # Per-target combined cloud: stack observer particles along the
        # particle axis with -inf log-weights for non-contributing rows.
        contrib_log_w = jnp.where(
            contributing[..., None],
            log_w,
            jnp.full_like(log_w, -jnp.inf),
        )
        combined_particles = jnp.transpose(particles, (1, 0, 2, 3)).reshape(n_total, n_obs * K, d)
        combined_log_w = jnp.transpose(contrib_log_w, (1, 0, 2)).reshape(n_total, n_obs * K)

        # Normalize.
        log_norm = jax.scipy.special.logsumexp(combined_log_w, axis=-1, keepdims=True)
        combined_log_w = combined_log_w - log_norm

        # Systematic resample per target down to K particles. We use
        # jax.random.fold_in over a fixed seed for determinism.
        base_key = jax.random.key(0)

        def _resample_k(target_idx, target_log_w, target_particles):
            key = jax.random.fold_in(base_key, target_idx)
            w = jnp.exp(target_log_w)
            cdf = jnp.cumsum(w)
            u0 = jax.random.uniform(key, (), minval=0.0, maxval=1.0 / K)
            u = u0 + jnp.arange(K) / K
            idx = jnp.searchsorted(cdf, u)
            return target_particles[idx]

        resampled = jax.vmap(_resample_k, in_axes=(0, 0, 0))(
            jnp.arange(n_total), combined_log_w, combined_particles
        )  # (N_total, K, d)

        new_particles = jnp.broadcast_to(resampled[None, ...], particles.shape)
        # Uniform log-weights post-resample.
        new_log_w = jnp.zeros_like(log_w) - jnp.log(K)

        # Apply only on (in-contact observer, opposing target) cells.
        apply_mask_p = contributing[..., None, None]
        apply_mask_w = contributing[..., None]

        new_particles = jnp.where(apply_mask_p, new_particles, particles)
        new_log_w = jnp.where(apply_mask_w, new_log_w, log_w)

        return belief.replace(particles=new_particles, log_weights=new_log_w)
