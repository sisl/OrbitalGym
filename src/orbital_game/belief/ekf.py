"""EKF belief + extended Kalman updater.

Belief shape mirrors KFBelief: (N_obs, N_total, d). EKFBeliefUpdater computes
state-transition Jacobians via jax.jacfwd over the user-supplied dynamics_fn,
and observation Jacobians via jax.jacfwd over Observation.obs_fn (when set).
For Observation channels with obs_fn=None, the linear obs_matrix is used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from orbital_game.belief._common import (
    _truth_arrays_for_side,
    build_initial_mean_from_truth,
    build_uniform_cov,
    build_uniform_mean,
)
from orbital_game.belief.kf import _single_pair_correct
from orbital_game.env.types import Side
from orbital_game.observations.types import Observation
from orbital_game.registry import BeliefInitializerKey, BeliefUpdaterKey, register


@flax.struct.dataclass
class EKFBelief:
    mean: jax.Array  # (N_obs, N_total, d)
    cov: jax.Array  # (N_obs, N_total, d, d)


@register(BeliefUpdaterKey.EKF)
@dataclass(frozen=True)
class EKFBeliefUpdater:
    """Extended Kalman filter with autodiff Jacobians.

    `dynamics_fn(x, u, dt) -> x_next` is the per-target nonlinear propagator.
    Per-pair Jacobian F_ik = jax.jacfwd(dynamics_fn)(mean[i,k]) at the current mean.
    For Observation channels with obs_fn set, H_ik = jax.jacfwd(obs_fn)(predicted_mean[i,k]).
    """

    dynamics_fn: Callable
    process_noise: jax.Array
    dt: float
    use_joseph_form: bool = True

    def __call__(
        self,
        belief: EKFBelief,
        observations: tuple[Observation, ...],
        action: jax.Array,
        side: Side,
        key: jax.Array,
    ) -> EKFBelief:
        del side, key
        n_obs, n_total, d = belief.mean.shape
        del d

        eye = jnp.eye(n_obs, n_total)
        per_pair_action = action[:, None, :] * eye[:, :, None]  # (n_obs, n_total, u_dim)

        def predict_pair(mean_ik, cov_ik, u_ik):
            new_mean = self.dynamics_fn(mean_ik, u_ik, self.dt)
            F = jax.jacfwd(lambda x: self.dynamics_fn(x, u_ik, self.dt))(mean_ik)  # noqa: N806
            new_cov = F @ cov_ik @ F.T + self.process_noise
            return new_mean, new_cov

        predict_per_pair = jax.vmap(jax.vmap(predict_pair, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
        mean_pred, cov_pred = predict_per_pair(belief.mean, belief.cov, per_pair_action)

        mean_cur, cov_cur = mean_pred, cov_pred
        for ch in observations:
            mean_cur, cov_cur = self._apply_channel(mean_cur, cov_cur, ch)

        # pyrefly: ignore[bad-argument-type]
        return EKFBelief(mean=mean_cur, cov=cov_cur)

    def _apply_channel(self, mean, cov, ch: Observation):
        if ch.obs_fn is None:
            per_pair = jax.vmap(
                jax.vmap(
                    lambda m, c, o: _single_pair_correct(
                        m, c, o, ch.obs_matrix, ch.obs_noise, self.use_joseph_form
                    ),
                    in_axes=(0, 0, 0),
                ),
                in_axes=(0, 0, 0),
            )
            new_mean, new_cov = per_pair(mean, cov, ch.obs)
        else:
            obs_fn = ch.obs_fn

            def correct_pair_nonlinear(m, c, o):
                H = jax.jacfwd(obs_fn)(m)  # noqa: N806
                S = H @ c @ H.T + ch.obs_noise  # noqa: N806
                K = c @ H.T @ jnp.linalg.inv(S)  # noqa: N806
                innovation = o - obs_fn(m)
                new_mean_ik = m + K @ innovation
                identity = jnp.eye(c.shape[0])
                if self.use_joseph_form:
                    i_minus_kh = identity - K @ H
                    new_cov_ik = i_minus_kh @ c @ i_minus_kh.T + K @ ch.obs_noise @ K.T
                else:
                    new_cov_ik = (identity - K @ H) @ c
                return new_mean_ik, new_cov_ik

            per_pair = jax.vmap(
                jax.vmap(correct_pair_nonlinear, in_axes=(0, 0, 0)),
                in_axes=(0, 0, 0),
            )
            new_mean, new_cov = per_pair(mean, cov, ch.obs)

        visible_mean = ch.visible[:, :, None]
        visible_cov = ch.visible[:, :, None, None]
        out_mean = jnp.where(visible_mean, new_mean, mean)
        out_cov = jnp.where(visible_cov, new_cov, cov)
        return out_mean, out_cov  # pyrefly: ignore[bad-return]


@register(BeliefInitializerKey.EKF_FROM_TRUTH)
@dataclass(frozen=True)
class EKFFromTruthInitializer:
    layout: Any
    variance_diag: jax.Array

    def __call__(self, env_state, side: Side, key) -> EKFBelief:
        del key
        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        mean = build_initial_mean_from_truth(own_truth, opp_truth)
        n_obs, n_total, _ = mean.shape
        cov = build_uniform_cov(n_obs, n_total, self.variance_diag)
        return EKFBelief(mean=mean, cov=cov)


@register(BeliefInitializerKey.EKF_UNIFORM_DEFAULT)
@dataclass(frozen=True)
class EKFUniformDefaultInitializer:
    layout: Any
    default_mean: jax.Array
    variance_diag: jax.Array

    def __call__(self, env_state, side: Side, key) -> EKFBelief:
        del env_state, key
        n_self = self.layout.n_guards if side is Side.GUARD else self.layout.n_bandits
        n_tgt = self.layout.n_bandits if side is Side.GUARD else self.layout.n_guards
        n_total = n_self + n_tgt
        mean = build_uniform_mean(n_self, n_total, self.default_mean)
        cov = build_uniform_cov(n_self, n_total, self.variance_diag)
        return EKFBelief(mean=mean, cov=cov)
