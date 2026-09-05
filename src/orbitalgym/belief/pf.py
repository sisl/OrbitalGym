"""Particle-filter belief + sequential-importance-resampling updater.

Belief shape mirrors KFBelief / EKFBelief at the per-(observer, target)
level, with an additional particle axis ``K``:

  particles:       (N_obs, N_total, K, d)
  log_weights:     (N_obs, N_total, K)   — log-softmax-normalized
  n_eff:           (N_obs, N_total)      — pre-resample effective sample size
  weight_entropy:  (N_obs, N_total)      — pre-resample Shannon entropy of weights
  resampled:       (N_obs, N_total) bool — did this pair resample this step?

``n_eff`` and ``weight_entropy`` are *pre-resample* — they describe the
posterior concentration produced by the latest measurement update, i.e.
the **collapse** signal. Post-resample weights are by construction
uniform and would mask the collapse if recorded instead. ``resampled``
flags exactly which pairs were just collapsed-and-redrawn this step,
so a `belief_rollout` history exposes when each detection event
happened.

The ``mean`` attribute on `Belief` is satisfied as a ``@property``
computing the weighted mean over the K axis. This makes a particle
filter fully interchangeable with KF/EKF in `belief_rollout` — policies
that read ``belief.mean`` get the same shape ``(N_obs, N_total, d)``.

Designed for scenarios where the prior is *non-Gaussian* — most notably
the LBG ring-intercept setup, where the bandit's initial position is
known to be on a 2:1 RT-plane natural-motion ellipse but uniformly
distributed in phase. A Kalman filter would collapse that prior onto
the lady; a particle filter expresses it directly and only collapses
once the range-limited sensor produces a real measurement.

Updater pipeline per step:

  1. PROPAGATE — push every particle through the user's
     ``dynamics_fn(x, u, dt)``. Self-pair particles get the ego action
     (k == i, via the same ``eye[i, k]`` mask used by KF / EKF);
     cross-pairs get zero action since we do not know what the
     opponent is commanding.
  2. PROPOSE — draw the new particle cloud from the measurement-
     conditioned proposal ``p(x_t | f(x_{t-1}), z_t)``, formed from the
     pair's prior covariance ``P`` and the linear (``obs_fn=None``)
     channels visible for the pair, and accumulate the matching
     importance weight ``log N(z; H f(x), H P Hᵀ + R)``. With no
     visible linear channel the proposal collapses to ``N(f(x), Q)``
     and the weight increment to zero, i.e. a plain bootstrap predict.
  3. UPDATE — fold each nonlinear channel's ``log N(z; obs_fn(x), R)``
     on its visible pairs, then each gated channel's negative-
     information term on its non-visible pairs.
  4. METRICS — compute pre-resample ``n_eff`` and ``weight_entropy``.
  5. RESAMPLE — per-pair systematic resampling fires when
     ``N_eff / K < n_eff_threshold``. Resampled particles get uniform
     log-weights and an optional jitter draw to combat sample
     impoverishment.

Step 2 is what keeps a directly measured cell — an onboard-GPS own
state, or an opponent inside a narrow cone — pinned to its measurement.
A bootstrap proposal only reaches a few ``sqrt(Q)`` per step, so once a
cell's error grows past that the measurement lands in the tail of every
particle, one arbitrary particle takes all the weight, and the cell runs
away instead of recovering. Sampling from the conditioned proposal
places the cloud on the measurement in one step no matter how far the
prior had drifted.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from orbitalgym.belief._common import _truth_arrays_for_side
from orbitalgym.env.types import Side
from orbitalgym.observations.negative_info import Hard, NegativeInfoMode, Off, Soft
from orbitalgym.observations.types import Observation
from orbitalgym.registry import BeliefInitializerKey, BeliefUpdaterKey, register

# log(1e-12) — used as the "crushing" log-weight in Hard mode (and as a
# floor in Soft mode if the score is large enough to saturate). Finite
# (≈ -27.6) so resampling's softmax+cumsum can't produce NaN even on a
# fully saturated cloud.
LOG_EPS = float(jnp.log(jnp.asarray(1e-12)))


@flax.struct.dataclass
class ParticleFilterBelief:
    """Particle-cloud belief, per (observer, target) pair.

    ``particles`` and ``log_weights`` carry the same leading
    ``(N_obs, N_total, K)`` indices; ``n_eff``, ``weight_entropy``, and
    ``resampled`` carry ``(N_obs, N_total)``. ``log_weights`` is stored
    already log-softmax-normalized along the particle (last) axis so
    taking ``softmax`` produces a valid probability vector at any moment.

    The three pair-level metrics are *pre-resample* — they describe the
    posterior produced by the most recent measurement update. After a
    resample fires for a pair, weights are reset to uniform; storing the
    pre-resample values is what makes the collapse curve visible across
    a belief history (``softmax(log_weights)`` after the resample is
    uniform by construction).
    """

    particles: jax.Array  # (..., N_obs, N_total, K, d)
    log_weights: jax.Array  # (..., N_obs, N_total, K)
    n_eff: jax.Array  # (..., N_obs, N_total)        — pre-resample effective sample size
    weight_entropy: jax.Array  # (..., N_obs, N_total)        — pre-resample Shannon entropy
    resampled: jax.Array  # (..., N_obs, N_total) bool   — fired this step?

    @property
    def mean(self) -> jax.Array:
        """Weighted mean over the particle axis. Shape (..., N_obs, N_total, d)."""
        # Use ``...`` so this property works whether the belief is the
        # per-step shape (N_obs, N_total, K, d) or has a leading time
        # axis from a `belief_rollout` scan.
        weights = jax.nn.softmax(self.log_weights, axis=-1)
        return jnp.einsum("...k,...kd->...d", weights, self.particles)


def _gaussian_log_likelihood(residual: jax.Array, R: jax.Array) -> jax.Array:  # noqa: N803
    """Multivariate-normal log-pdf up to a constant in ``R``.

    Drops the ``-0.5 * log det(2π R)`` term — it cancels under
    log-softmax across particles since R is the same for every particle
    within a pair, so only the Mahalanobis term affects the resulting
    weights. A per-pair R varies across pairs, not across the particles
    the softmax normalizes over, so the cancellation still holds. Keeping
    the residual term alone is faster and avoids a `slogdet` call per pair.
    """
    solved = jnp.linalg.solve(R, residual)
    return -0.5 * residual @ solved


# Relative floor added to the unit-diagonal form of a covariance before it is
# factorized or solved. Sized for float32: the scenarios that drive this filter
# mix position and velocity in one state, so a covariance can span seven orders
# of magnitude and an absolute floor is either negligible at the top of that
# range or dominant at the bottom.
_PSD_FLOOR = 1e-6
_DIAG_FLOOR = 1e-30


def _diag_scale(a: jax.Array) -> jax.Array:
    """Per-dimension scale ``sqrt(diag(a))``, floored away from zero."""
    return jnp.sqrt(jnp.clip(jnp.diagonal(a), _DIAG_FLOOR))


def _psd_cholesky(a: jax.Array) -> jax.Array:
    """Lower Cholesky factor of a symmetric PSD matrix.

    Factorizes the unit-diagonal (correlation) form and rescales, so the
    relative floor is meaningful in every dimension whatever the spread
    of scales along the diagonal. Factorizing the raw matrix in float32
    fails outright once its condition number passes ``1/eps``: the
    smallest pivot rounds negative and the factorization returns NaN.
    """
    scale = _diag_scale(a)
    outer = scale[:, None] * scale[None, :]
    unit = a / outer
    unit = 0.5 * (unit + unit.T) + _PSD_FLOOR * jnp.eye(a.shape[0], dtype=a.dtype)
    return scale[:, None] * jnp.linalg.cholesky(unit)


def _psd_solve(a: jax.Array, b: jax.Array) -> jax.Array:
    """Solve ``a x = b`` for a symmetric PSD ``a``, preconditioned by its diagonal."""
    scale = _diag_scale(a)
    outer = scale[:, None] * scale[None, :]
    unit = a / outer
    unit = 0.5 * (unit + unit.T) + _PSD_FLOOR * jnp.eye(a.shape[0], dtype=a.dtype)
    return jnp.linalg.solve(unit, b / scale[:, None]) / scale[:, None]


def _stack_linear_channels(
    channels: tuple[Observation, ...], i: jax.Array, j: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Stacked ``(H, z, R, any_visible)`` for one (observer, target) pair.

    A channel that is not visible for the pair contributes a zero ``H``
    block, a zero measurement and an identity ``R`` block, which leaves
    both the conditioned proposal and its importance weight untouched.
    ``any_visible`` is True when at least one channel is visible.
    """
    matrices, measurements, noises, visibilities = [], [], [], []
    for ch in channels:
        m = ch.obs.shape[-1]
        visible = ch.visible[i, j]
        visibilities.append(visible)
        matrices.append(jnp.where(visible, ch.obs_matrix, jnp.zeros_like(ch.obs_matrix)))
        measurements.append(jnp.where(visible, ch.obs[i, j], jnp.zeros((m,), dtype=ch.obs.dtype)))
        noises.append(jnp.where(visible, ch.noise_for(i, j), jnp.eye(m, dtype=ch.obs_matrix.dtype)))
    return (
        jnp.concatenate(matrices, axis=0),
        jnp.concatenate(measurements, axis=0),
        jax.scipy.linalg.block_diag(*noises),
        jnp.any(jnp.stack(visibilities)),
    )


@register(BeliefUpdaterKey.PF)
@dataclass(frozen=True)
class ParticleFilterBeliefUpdater:
    """Sequential-importance-resampling particle filter.

    Parameters
    ----------
    dynamics_fn:
        Per-vehicle propagator with signature ``(x, u, dt) -> x_next``.
        Same shape contract as ``EKFBeliefUpdater.dynamics_fn``.
    process_noise:
        ``(d, d)`` covariance ``Q`` used to draw additive Gaussian
        process noise during predict (and resample jitter). Pass a tiny
        positive-definite matrix when you want near-deterministic
        propagation; pure zeros are allowed but make resample-jitter a
        no-op.
    dt:
        Step size in seconds passed to ``dynamics_fn``.
    n_eff_threshold:
        Resample fires per-pair when ``N_eff / K`` falls below this
        fraction. ``0.5`` is the standard default; ``1.0`` resamples
        every step, ``0.0`` never resamples.
    resample_jitter_scale:
        After resampling, add ``scale * chol(Q) * randn`` to particles
        so duplicate samples differ slightly. Default ``0.0`` (no
        jitter). Useful when the measurement is sharp and resamples
        produce many copies of the same particle.
    """

    dynamics_fn: Callable
    process_noise: jax.Array  # (d, d)
    dt: float
    n_eff_threshold: float = 0.5
    resample_jitter_scale: float = 0.0
    negative_info: NegativeInfoMode = field(default_factory=Off)

    def __call__(
        self,
        belief: ParticleFilterBelief,
        observations: tuple[Observation, ...],
        action: jax.Array,  # (N_obs, u_dim)
        side: Side,
        key: jax.Array,
    ) -> ParticleFilterBelief:
        del side
        # ---- VALIDATE negative-info config against observation channels ----
        if not isinstance(self.negative_info, Off):
            gated_indices = [
                i for i, ch in enumerate(observations) if ch.visibility_score_fn is not None
            ]
            if not gated_indices:
                raise ValueError(
                    f"negative_info={type(self.negative_info).__name__}() requires at least one "
                    "observation channel with a visibility_score_fn, but no observation channel "
                    "supplied one. Either switch to Off() or use a gated sensor "
                    "(RangeLimitedObservation, ConicalObservation)."
                )
        if isinstance(self.negative_info, Soft):
            n_chan = len(observations)
            n_soft = len(self.negative_info.softness_per_channel)
            if n_soft != n_chan:
                raise ValueError(
                    f"Soft.softness_per_channel has {n_soft} entries but {n_chan} observation "
                    "channels were supplied. Provide one positive softness per channel "
                    "(non-gated channel entries are ignored but required for positional alignment)."
                )
        n_obs, n_total, k_particles, d = belief.particles.shape

        k_predict, k_update, k_resample = jax.random.split(key, 3)
        del k_update  # weight update is deterministic given particles + obs

        # ---- PREDICT ----
        # Apply ego action only to self-pairs; opponent pairs propagate freely.
        u_dim = action.shape[-1]
        eye = jnp.eye(n_obs, n_total)  # (n_obs, n_total)
        u_per_pair = action[:, None, :] * eye[:, :, None]  # (n_obs, n_total, u_dim)
        u_per_particle = jnp.broadcast_to(
            u_per_pair[:, :, None, :],
            (n_obs, n_total, k_particles, u_dim),
        )

        def step_one(x: jax.Array, u: jax.Array) -> jax.Array:
            return self.dynamics_fn(x, u, self.dt)

        step_per_particle = jax.vmap(jax.vmap(jax.vmap(step_one)))
        propagated = step_per_particle(belief.particles, u_per_particle)

        # Regularized Q keeps the proposal covariance factorizable with a zero
        # process noise, where the proposal degenerates to the deterministic
        # propagation.
        q_reg = self.process_noise + 1e-12 * jnp.eye(d)
        chol_q = jnp.linalg.cholesky(q_reg)

        # ---- PROPOSE ----
        predicted, log_w = self._propose(
            propagated, belief.log_weights, observations, q_reg, chol_q, k_predict
        )
        log_w = belief.log_weights + log_w

        # ---- UPDATE (sequential per channel) ----
        for i, ch in enumerate(observations):
            log_w = self._apply_channel(predicted, log_w, ch, i)

        # Normalize once after folding all channels.
        log_w = jax.nn.log_softmax(log_w, axis=-1)

        # ---- METRICS (pre-resample) ----
        # softmax(log_softmax(x)) == softmax(x); compute weights once and
        # reuse for n_eff, entropy, and resample.
        weights = jnp.exp(log_w)  # already normalized to sum = 1 along axis=-1
        n_eff = 1.0 / jnp.sum(weights * weights, axis=-1)  # (N_obs, N_total)
        # Use weights * log_w directly (log_w is already normalized) to
        # avoid log(0) when a particle weight is exactly zero — that
        # particle's contribution is 0 regardless of log_w by L'Hopital.
        weight_entropy = -jnp.sum(weights * log_w, axis=-1)  # (N_obs, N_total)
        resampled_mask = (n_eff / k_particles) < self.n_eff_threshold

        # ---- RESAMPLE ----
        new_particles, new_log_w = self._apply_resample(
            predicted, log_w, weights, resampled_mask, chol_q, k_resample
        )

        return ParticleFilterBelief(
            particles=new_particles,
            log_weights=new_log_w,
            n_eff=n_eff,
            weight_entropy=weight_entropy,
            resampled=resampled_mask,
        )

    def _propose(
        self,
        propagated: jax.Array,  # (N_obs, N_total, K, d)
        prior_log_w: jax.Array,  # (N_obs, N_total, K)
        observations: tuple[Observation, ...],
        q_reg: jax.Array,  # (d, d)
        chol_q: jax.Array,  # (d, d)
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Draw the new cloud from ``p(x | f(x_prev), z)`` and weight it.

        Stacks every linear channel into one measurement per pair, with
        invisible channels neutralized to a zero ``H`` row block, a zero
        residual and an identity ``R`` block so they move neither the
        proposal mean nor the weight.

        The prior each pair is conditioned on is the kernel-density view
        of its predicted cloud: every particle carries covariance ``P =
        h² C + Q``, where ``C`` is the cloud's weighted covariance and
        ``h`` is Silverman's bandwidth for ``K`` samples in ``d``
        dimensions. Conditioning on ``Q`` alone would move each particle
        by at most a fraction of one step's process noise, so a
        measurement sharper than a cloud that is broad for any other
        reason — a ring prior, drift accumulated while unseen — lands in
        the tail of every particle, one arbitrary particle takes the
        whole weight, and the resample throws the remaining spread away.
        Conditioning on ``P`` repositions the cloud onto such a
        measurement instead, and leaves a tight cloud (where ``C`` is
        already of order ``Q``) essentially unchanged.

        Pairs with nothing visible keep the plain ``N(f(x), Q)`` draw and
        a zero weight increment: kernel-smoothing an unmeasured cloud
        would inflate it every step.

        Returns the drawn particles and the per-particle log-weight
        increment, both shaped like the inputs.
        """
        n_obs, n_total, k_particles, d = propagated.shape
        linear = tuple(ch for ch in observations if ch.obs_fn is None)
        if not linear:
            draw = jax.random.normal(key, propagated.shape) @ chol_q.T
            return propagated + draw, jnp.zeros((n_obs, n_total, k_particles))

        bandwidth_sq = (4.0 / (k_particles * (d + 2))) ** (2.0 / (d + 4))
        prior_w = jax.nn.softmax(prior_log_w, axis=-1)

        def one_pair(
            xf: jax.Array, w: jax.Array, i: jax.Array, j: jax.Array, pair_key: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            H, z, R, any_visible = _stack_linear_channels(linear, i, j)  # noqa: N806
            centered = xf - w @ xf
            cloud_cov = (w[:, None] * centered).T @ centered
            P = jnp.where(any_visible, bandwidth_sq * cloud_cov + q_reg, q_reg)  # noqa: N806
            S = H @ P @ H.T + R  # noqa: N806
            gain = _psd_solve(S, H @ P).T  # (d, M) — P Hᵀ S⁻¹
            residual = z[None, :] - xf @ H.T  # (K, M)
            shifted = xf + residual @ gain.T  # (K, d)

            # Sample the conditioned covariance as ``u - gain (H u + v)``
            # for ``u ~ N(0, P)`` and ``v ~ N(0, R)``. That has exactly the
            # conditioned covariance ``(I - gain H) P (I - gain H)ᵀ +
            # gain R gainᵀ`` without ever forming it: subtracting the
            # measurement's information from ``P`` cancels away most of the
            # significant digits when the cloud is broad relative to the
            # measurement, and the difference is then as likely to
            # factorize as not.
            k_state, k_meas = jax.random.split(pair_key, 2)
            u = jax.random.normal(k_state, (k_particles, d), dtype=xf.dtype) @ _psd_cholesky(P).T
            v = (
                jax.random.normal(k_meas, (k_particles, R.shape[0]), dtype=xf.dtype)
                @ _psd_cholesky(R).T
            )
            drawn = shifted + u - (u @ H.T + v) @ gain.T

            # Mahalanobis term only: log det S is constant across the
            # particles the per-pair log-softmax normalizes over.
            log_inc = -0.5 * jnp.sum(residual * _psd_solve(S, residual.T).T, axis=-1)
            return drawn, log_inc

        keys = jax.random.split(key, n_obs * n_total).reshape(n_obs, n_total, 2)
        per_target = jax.vmap(one_pair, in_axes=(0, 0, None, 0, 0))
        per_observer = jax.vmap(per_target, in_axes=(0, 0, 0, None, 0))
        return per_observer(propagated, prior_w, jnp.arange(n_obs), jnp.arange(n_total), keys)

    def _apply_channel(
        self,
        particles: jax.Array,  # (N_obs, N_total, K, d)
        log_w: jax.Array,  # (N_obs, N_total, K)
        ch: Observation,
        channel_index: int,
    ) -> jax.Array:
        """Fold one channel's nonlinear likelihood and negative information.

        Linear channels are already accounted for by the proposal, so only
        their negative-information term is left to apply here.
        """
        visible = ch.visible[:, :, None]

        if ch.obs_fn is not None:
            obs_fn = ch.obs_fn

            def lik_one(p: jax.Array, z: jax.Array, R: jax.Array) -> jax.Array:  # noqa: N803
                return _gaussian_log_likelihood(z - obs_fn(p), R)

            per_particle = jax.vmap(lik_one, in_axes=(0, None, None))

            def lik_one_pair(p: jax.Array, z: jax.Array, i: jax.Array, j: jax.Array) -> jax.Array:
                return per_particle(p, z, ch.noise_for(i, j))

            n_obs_axis, n_total_axis = ch.visible.shape
            per_target = jax.vmap(lik_one_pair, in_axes=(0, 0, None, 0))
            per_observer = jax.vmap(per_target, in_axes=(0, 0, 0, None))
            log_lik = per_observer(
                particles, ch.obs, jnp.arange(n_obs_axis), jnp.arange(n_total_axis)
            )  # (N_obs, N_total, K)
            log_w = log_w + jnp.where(visible, log_lik, 0.0)

        if ch.visibility_score_fn is None or isinstance(self.negative_info, Off):
            return log_w

        score = ch.visibility_score_fn(particles)  # (N_obs, N_total, K)
        if isinstance(self.negative_info, Hard):
            log_p_no_detect = jnp.where(score > 0, LOG_EPS, 0.0)
        elif isinstance(self.negative_info, Soft):
            softness = self.negative_info.softness_per_channel[channel_index]
            log_p_no_detect = jax.nn.log_sigmoid(-score / softness)
        else:
            raise TypeError(f"Unsupported NegativeInfoMode: {type(self.negative_info).__name__}")
        return log_w + jnp.where(visible, 0.0, log_p_no_detect)

    def _apply_resample(
        self,
        particles: jax.Array,  # (N_obs, N_total, K, d)
        log_w: jax.Array,  # (N_obs, N_total, K)
        weights: jax.Array,  # (N_obs, N_total, K)  — already softmax-normalized
        do_resample: jax.Array,  # (N_obs, N_total) bool — precomputed mask
        chol_q: jax.Array,  # (d, d)
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        n_obs, n_total, k_particles, _ = particles.shape

        # ---- Systematic resampling, vmapped over (observer, target) ----
        def systematic_one_pair(parts: jax.Array, w: jax.Array, k: jax.Array) -> jax.Array:
            cumsum = jnp.cumsum(w)
            u0 = jax.random.uniform(k, ())
            us = (u0 + jnp.arange(k_particles, dtype=cumsum.dtype)) / k_particles
            idx = jnp.searchsorted(cumsum, us, side="right")
            idx = jnp.clip(idx, 0, k_particles - 1)
            return parts[idx]

        k_split, k_jitter = jax.random.split(key, 2)
        keys_per_obs = jax.random.split(k_split, n_obs * n_total).reshape(n_obs, n_total, 2)
        per_target_resample = jax.vmap(systematic_one_pair, in_axes=(0, 0, 0))
        per_observer_resample = jax.vmap(per_target_resample, in_axes=(0, 0, 0))
        resampled = per_observer_resample(particles, weights, keys_per_obs)

        if self.resample_jitter_scale > 0.0:
            jitter = jax.random.normal(k_jitter, resampled.shape) @ chol_q.T
            resampled = resampled + self.resample_jitter_scale * jitter

        # Apply only to pairs whose N_eff dipped below threshold.
        do_resample_p = do_resample[:, :, None, None]
        new_particles = jnp.where(do_resample_p, resampled, particles)

        # Resampled pairs reset to uniform weights; the others keep their
        # current normalized weights.
        do_resample_w = do_resample[:, :, None]
        uniform_log_w = jnp.full(log_w.shape, -jnp.log(jnp.asarray(k_particles, dtype=log_w.dtype)))
        new_log_w = jnp.where(do_resample_w, uniform_log_w, log_w)
        return new_particles, new_log_w


# ---- Initializers --------------------------------------------------------


def _uniform_log_weights(n_obs: int, n_total: int, k_particles: int) -> jax.Array:
    return jnp.full((n_obs, n_total, k_particles), -jnp.log(float(k_particles)))


def _initial_metrics(
    n_obs: int, n_total: int, k_particles: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Initial pair-level metrics for a uniform-weight belief.

    n_eff = K, weight_entropy = log K, resampled = False.
    """
    n_eff = jnp.full((n_obs, n_total), float(k_particles))
    weight_entropy = jnp.full((n_obs, n_total), math.log(k_particles))
    resampled = jnp.zeros((n_obs, n_total), dtype=bool)
    return n_eff, weight_entropy, resampled


@register(BeliefInitializerKey.PF_FROM_TRUTH)
@dataclass(frozen=True)
class ParticleFilterFromTruthInitializer:
    """All particles start at truth, plus tiny isotropic jitter.

    Use when the belief should begin as a near-delta on the true
    state — e.g. the guard knows its own state exactly and starts with
    a perfect cheat-prior on the bandit. Jitter prevents singular
    weight updates downstream.
    """

    layout: Any
    n_particles: int
    jitter_scale: float = 1e-3  # std of isotropic Gaussian on each component

    def __call__(self, env_state, side: Side, key) -> ParticleFilterBelief:
        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        n_self = own_truth.shape[0]
        n_total = n_self + opp_truth.shape[0]
        d = self.layout.dynamics_state_dim
        stacked = jnp.concatenate([own_truth, opp_truth], axis=0)  # (n_total, d)
        # particles[i, k, p, :] = truth[k] + small jitter
        broadcast = jnp.broadcast_to(
            stacked[None, :, None, :], (n_self, n_total, self.n_particles, d)
        )
        jitter = self.jitter_scale * jax.random.normal(key, broadcast.shape)
        particles = broadcast + jitter
        log_w = _uniform_log_weights(n_self, n_total, self.n_particles)
        n_eff, weight_entropy, resampled = _initial_metrics(n_self, n_total, self.n_particles)
        return ParticleFilterBelief(
            particles=particles,
            log_weights=log_w,
            n_eff=n_eff,
            weight_entropy=weight_entropy,
            resampled=resampled,
        )


@register(BeliefInitializerKey.PF_RING)
@dataclass(frozen=True)
class ParticleFilterRingInitializer:
    """Uniform-on-bandit-ring prior for the LBG ring-intercept scenario.

    For each cross-pair (observer i, target k where k targets the
    *opposing* side), sample K particles uniformly along the 2:1
    RT-plane natural-motion ring at radius ``ring_radius_m``. Velocity
    components are set from the closed-form natural-motion solution so
    every particle is a valid free-drift trajectory under HCW dynamics
    — the cloud stays on the ring as it propagates.

    Self-pairs and same-side pairs are initialized to truth + tiny
    jitter so the observer's own-state estimate matches reality. The
    only ring-distributed pairs are (own-side observer, opposing-side
    target).

    Layout requirements: ``n_guards``, ``n_bandits``,
    ``dynamics_state_dim`` (must be 4 for RT plane or 6 for RTN). RTN
    places zero cross-track position and rate, creating an in-plane ring.

    Parameters
    ----------
    layout:
        ``cfg.layout`` from a built ``ScenarioConfig``.
    ring_radius_m:
        Radial-ellipse semi-major axis (matches the bandit's
        ``RelativeEllipse.radial_ellipse_m``). Along-track amplitude is
        2× this under HCW 2:1 motion.
    mean_motion_rad_s:
        Reference orbit mean motion (use
        ``orbitalgym.reference_orbit.mean_motion(cfg.reference_orbit)``).
    n_particles:
        Number of particles per (observer, target) pair.
    truth_jitter_scale:
        Jitter on self-pair / same-side particles (where particles are
        anchored to truth). Cross-pair particles do not get this
        jitter — their spread comes from the ring distribution itself.
    """

    layout: Any
    ring_radius_m: float
    mean_motion_rad_s: float
    n_particles: int
    truth_jitter_scale: float = 1e-3

    def __call__(self, env_state, side: Side, key) -> ParticleFilterBelief:
        d = self.layout.dynamics_state_dim
        if d not in (4, 6):
            raise ValueError(f"ParticleFilterRingInitializer expects d in (4, 6), got d={d}")

        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        n_self = own_truth.shape[0]
        n_opp = opp_truth.shape[0]
        n_total = n_self + n_opp

        k_phase, k_jitter = jax.random.split(key, 2)

        # Truth-anchored block for own-side targets (k < n_self).
        own_block = jnp.broadcast_to(
            own_truth[None, :, None, :], (n_self, n_self, self.n_particles, d)
        )
        own_jitter = self.truth_jitter_scale * jax.random.normal(k_jitter, own_block.shape)
        own_particles = own_block + own_jitter

        # Ring-distributed block for opposing-side targets (k >= n_self).
        # Phases sampled uniformly in [0, 2π) per (observer, target, particle).
        phase_shape = (n_self, n_opp, self.n_particles)
        phases = jax.random.uniform(k_phase, phase_shape, minval=0.0, maxval=2.0 * jnp.pi)
        cp = jnp.cos(phases)
        sp = jnp.sin(phases)
        a = self.ring_radius_m
        n_motion = self.mean_motion_rad_s
        # Closed-form natural-motion ring (matches sampling.side.RelativeEllipse,
        # along_track_offset_m=0, drift=0):
        #   r       = -a * cos(phase)
        #   t       =  2 a * sin(phase)
        #   r_dot   =  n a * sin(phase)
        #   t_dot   =  2 n a * cos(phase)
        opp_particles = jnp.stack(
            [
                -a * cp,
                2.0 * a * sp,
                n_motion * a * sp,
                2.0 * n_motion * a * cp,
            ],
            axis=-1,
        )  # (n_self, n_opp, n_particles, 4)

        if d == 6:
            zeros = jnp.zeros_like(cp)
            opp_particles = jnp.stack(
                [
                    -a * cp,
                    2.0 * a * sp,
                    zeros,
                    n_motion * a * sp,
                    2.0 * n_motion * a * cp,
                    zeros,
                ],
                axis=-1,
            )

        particles = jnp.concatenate([own_particles, opp_particles], axis=1)
        log_w = _uniform_log_weights(n_self, n_total, self.n_particles)
        n_eff, weight_entropy, resampled = _initial_metrics(n_self, n_total, self.n_particles)
        return ParticleFilterBelief(
            particles=particles,
            log_weights=log_w,
            n_eff=n_eff,
            weight_entropy=weight_entropy,
            resampled=resampled,
        )
