"""Gaussian belief + linear Kalman filter.

Predict and correct are exposed separately for testability; the combined
__call__ runs predict-then-correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.struct
import jax
import jax.numpy as jnp

from orbital_game.registry import BeliefUpdaterKey, register


@flax.struct.dataclass
class GaussianBelief:
    mean: jax.Array  # (D,) or (N_defenders, D) for decentralized
    cov: jax.Array  # (D, D) or (N_defenders, D, D)


@register(BeliefUpdaterKey.GAUSSIAN_KALMAN)
@dataclass(frozen=True)
class GaussianKalmanUpdater:
    """Linear Kalman filter.

    Predict: x' = F x + B u,   P' = F P Fᵀ + Q
    Correct: y = z - H x',     S = H P' Hᵀ + R, K = P' Hᵀ S⁻¹
             x'' = x' + K y
             P'' = (I - K H) P (I - K H)ᵀ + K R Kᵀ   (Joseph form, default)
             P'' = (I - K H) P                        (simple form, opt-in)

    All matrices are constants set at scenario build time (STM derived from
    planning_dynamics, B from actuator Jacobian, H from observation linearization,
    Q and R from user-supplied noise models).

    `use_joseph_form` defaults to True because the two forms are
    equivalent but Joseph structurally preserves symmetry and positive
    semi-definiteness under floating-point error accumulation over long rollouts.
    Pay the cost of two extra matrix multiplies per correction; get robust
    long-horizon behaviour in return.
    """

    stm: jax.Array  # F, (D, D)
    control_matrix: jax.Array  # B, (D, U)
    process_noise: jax.Array  # Q, (D, D)
    obs_matrix: jax.Array  # H, (M, D)
    obs_noise: jax.Array  # R, (M, M)
    use_joseph_form: bool = True

    def predict(self, belief: GaussianBelief, action: jax.Array) -> GaussianBelief:
        mean = self.stm @ belief.mean + self.control_matrix @ action
        cov = self.stm @ belief.cov @ self.stm.T + self.process_noise
        return GaussianBelief(mean=mean, cov=cov)

    def correct(self, belief: GaussianBelief, observation: jax.Array) -> GaussianBelief:
        H = self.obs_matrix  # noqa: N806
        R = self.obs_noise  # noqa: N806
        P = belief.cov  # noqa: N806
        S = H @ P @ H.T + R  # noqa: N806
        K = P @ H.T @ jnp.linalg.inv(S)  # noqa: N806
        innovation = observation - H @ belief.mean
        mean = belief.mean + K @ innovation
        identity = jnp.eye(P.shape[0])
        # Python bool on a frozen dataclass — static branch at trace time, no
        # dynamic branching in the jit-compiled path.
        if self.use_joseph_form:
            i_minus_kh = identity - K @ H
            cov = i_minus_kh @ P @ i_minus_kh.T + K @ R @ K.T
        else:
            cov = (identity - K @ H) @ P
        return GaussianBelief(mean=mean, cov=cov)

    def __call__(
        self,
        belief: GaussianBelief,
        action: jax.Array,
        observation: jax.Array,
        params,
        dt: float,
        key,
    ) -> GaussianBelief:
        del params, dt, key
        predicted = self.predict(belief, action)
        return self.correct(predicted, observation)
