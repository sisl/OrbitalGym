"""KF belief + linear Kalman updater (per-observer-per-target shape).

Belief shape:
  mean: (N_obs, N_total, d)
  cov:  (N_obs, N_total, d, d)

Updater consumes a tuple of Observation channels. One predict step is
applied to all (observer, tracked) pairs; B @ action[i] is added only for
the self-pair (k == i). Each channel's correction is then folded
sequentially, gated by the channel's per-pair `visible` mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from orbitalgym.belief._common import (
    _truth_arrays_for_side,
    build_initial_mean_from_truth,
    build_uniform_cov,
    build_uniform_mean,
)
from orbitalgym.env.types import Side
from orbitalgym.observations.types import Observation
from orbitalgym.registry import BeliefInitializerKey, BeliefUpdaterKey, register


@flax.struct.dataclass
class KFBelief:
    mean: jax.Array  # (N_obs, N_total, d)
    cov: jax.Array  # (N_obs, N_total, d, d)


def _single_pair_correct(
    mean: jax.Array,  # (d,)
    cov: jax.Array,  # (d, d)
    obs: jax.Array,  # (m,)
    H: jax.Array,  # (m, d)  # noqa: N803
    R: jax.Array,  # (m, m)  # noqa: N803
    use_joseph_form: bool,
) -> tuple[jax.Array, jax.Array]:
    S = H @ cov @ H.T + R  # noqa: N806
    K = cov @ H.T @ jnp.linalg.inv(S)  # noqa: N806
    innovation = obs - H @ mean
    new_mean = mean + K @ innovation
    identity = jnp.eye(cov.shape[0])
    if use_joseph_form:
        i_minus_kh = identity - K @ H
        new_cov = i_minus_kh @ cov @ i_minus_kh.T + K @ R @ K.T
    else:
        new_cov = (identity - K @ H) @ cov
    return new_mean, new_cov


@register(BeliefUpdaterKey.KF)
@dataclass(frozen=True)
class KFBeliefUpdater:
    """Linear Kalman filter, per-(observer, target) block-diagonal.

    `stm`, `control_matrix`, `process_noise` are uniform across all tracked
    entities. Per-pair correction matrices `H` and `R` come from each
    Observation channel (different sensors may have different m, H, R).
    """

    stm: jax.Array  # F, (d, d)
    control_matrix: jax.Array  # B, (d, u_dim)
    process_noise: jax.Array  # Q, (d, d)
    use_joseph_form: bool = True

    def __call__(
        self,
        belief: KFBelief,
        observations: tuple[Observation, ...],
        action: jax.Array,  # (N_obs, u_dim)
        side: Side,
        key: jax.Array,
    ) -> KFBelief:
        del side, key
        for ch in observations:
            if ch.obs_fn is not None:
                raise TypeError(
                    "KFBeliefUpdater rejects Observation channels with obs_fn set; "
                    "use EKFBeliefUpdater for nonlinear measurement models."
                )

        n_obs, n_total, d = belief.mean.shape
        F = self.stm  # noqa: N806
        B = self.control_matrix  # noqa: N806
        Q = self.process_noise  # noqa: N806

        # ---- Predict ----
        # mean_pred[i, k] = F @ mean[i, k]  (apply F to each (i, k) state vector)
        # @ broadcasts: (n_obs, n_total, d) @ (d, d).T → (n_obs, n_total, d)
        mean_pred = belief.mean @ F.T
        # B @ action[i] applies only to self-pair (k == i).
        bu_per_obs = action @ B.T  # (n_obs, d)
        eye = jnp.eye(n_obs, n_total)  # (n_obs, n_total) — 1 at (i, i), 0 else
        mean_pred = mean_pred + bu_per_obs[:, None, :] * eye[:, :, None]

        # cov_pred[i, k] = F @ cov[i, k] @ F.T + Q
        cov_pred = F @ belief.cov @ F.T + Q

        # ---- Correct (one channel at a time) ----
        mean_cur, cov_cur = mean_pred, cov_pred
        for ch in observations:
            mean_cur, cov_cur = self._apply_channel(mean_cur, cov_cur, ch)

        return KFBelief(mean=mean_cur, cov=cov_cur)

    def _apply_channel(
        self,
        mean: jax.Array,  # (N_obs, N_total, d)
        cov: jax.Array,  # (N_obs, N_total, d, d)
        ch: Observation,
    ) -> tuple[jax.Array, jax.Array]:
        def _pair(
            m: jax.Array, c: jax.Array, o: jax.Array, i: jax.Array, j: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return _single_pair_correct(
                m, c, o, ch.obs_matrix, ch.noise_for(i, j), self.use_joseph_form
            )

        n_obs, n_total = ch.visible.shape
        per_pair = jax.vmap(jax.vmap(_pair, in_axes=(0, 0, 0, None, 0)), in_axes=(0, 0, 0, 0, None))
        new_mean, new_cov = per_pair(mean, cov, ch.obs, jnp.arange(n_obs), jnp.arange(n_total))
        visible_mean = ch.visible[:, :, None]  # (N_obs, N_total, 1)
        visible_cov = ch.visible[:, :, None, None]  # (N_obs, N_total, 1, 1)
        out_mean = jnp.where(visible_mean, new_mean, mean)
        out_cov = jnp.where(visible_cov, new_cov, cov)
        return out_mean, out_cov  # pyrefly: ignore[bad-return]


@register(BeliefInitializerKey.KF_FROM_TRUTH)
@dataclass(frozen=True)
class KFFromTruthInitializer:
    """Initial belief: per-observer mean = truth of every tracked entity.

    Layout requirements: must expose `n_guards`, `n_bandits`,
    `dynamics_state_dim` attributes.
    """

    layout: Any  # has n_guards, n_bandits, dynamics_state_dim
    variance_diag: jax.Array  # (d,)

    def __call__(self, env_state, side: Side, key) -> KFBelief:
        del key
        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        mean = build_initial_mean_from_truth(own_truth, opp_truth)
        n_obs, n_total, d = mean.shape
        cov = build_uniform_cov(n_obs, n_total, self.variance_diag)
        return KFBelief(mean=mean, cov=cov)


@register(BeliefInitializerKey.KF_UNIFORM_DEFAULT)
@dataclass(frozen=True)
class KFUniformDefaultInitializer:
    """Uninformed initial belief: every (observer, tracked) pair gets the same prior."""

    layout: Any  # has n_guards, n_bandits
    default_mean: jax.Array  # (d,)
    variance_diag: jax.Array  # (d,)

    def __call__(self, env_state, side: Side, key) -> KFBelief:
        del env_state, key
        n_self = self.layout.n_guards if side is Side.GUARD else self.layout.n_bandits
        n_tgt = self.layout.n_bandits if side is Side.GUARD else self.layout.n_guards
        n_total = n_self + n_tgt
        mean = build_uniform_mean(n_self, n_total, self.default_mean)
        cov = build_uniform_cov(n_self, n_total, self.variance_diag)
        return KFBelief(mean=mean, cov=cov)
