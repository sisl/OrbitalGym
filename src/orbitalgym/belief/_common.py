"""Shared helpers for KF and EKF belief construction.

Centralizes the per-(observer, tracked) shape so KFBelief and EKFBelief
initializers don't drift apart.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _truth_arrays_for_side(env_state: Any, side: str) -> tuple[jax.Array, jax.Array]:
    """Return (own_truth, opposing_truth) raw dynamics-state arrays for `side`.

    Reads `.rtn` (the dynamics-state leaf wrapped by RTNState) — the belief
    only tracks the dynamics state, not mass/power/attitude. If a side's
    state component bundle does not include RTN, falls back to `.rt`.
    """
    own_state = env_state.guards if side == "guard" else env_state.bandits
    opp_state = env_state.bandits if side == "guard" else env_state.guards

    def _extract(s):
        if hasattr(s, "rtn"):
            return s.rtn
        if hasattr(s, "rt"):
            return s.rt
        raise AttributeError("State has no dynamics component (RTState or RTNState)")

    return _extract(own_state), _extract(opp_state)


def build_initial_mean_from_truth(
    own_truth: jax.Array,  # (N_self, d)
    opp_truth: jax.Array,  # (N_tgt, d)
) -> jax.Array:
    """Build (N_obs, N_total, d) mean from per-side truth arrays.

    For every observer i in the own side, populate mean[i, k] with truth
    of entity k. Own-side entities come first (k < N_self); opposing-side
    entities come second (k >= N_self).
    """
    n_self = own_truth.shape[0]
    stacked = jnp.concatenate([own_truth, opp_truth], axis=0)  # (N_total, d)
    return jnp.broadcast_to(stacked[None, :, :], (n_self, *stacked.shape))


def build_uniform_mean(
    n_obs: int,
    n_total: int,
    default_mean: jax.Array,  # (d,)
) -> jax.Array:
    """Build (N_obs, N_total, d) by tiling a single default mean."""
    return jnp.broadcast_to(default_mean[None, None, :], (n_obs, n_total, default_mean.shape[0]))


def build_uniform_cov(
    n_obs: int,
    n_total: int,
    variance_diag: jax.Array,  # (d,)
) -> jax.Array:
    """Build (N_obs, N_total, d, d) by tiling a diagonal cov."""
    d = variance_diag.shape[0]
    cov = jnp.diag(variance_diag)
    return jnp.broadcast_to(cov[None, None, :, :], (n_obs, n_total, d, d))
