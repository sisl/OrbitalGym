"""TeammateEphemerisObservation — cooperative ephemeris sharing between own-side vehicles.

Each observer i gets a noisy reading of every other own-side vehicle's full
dynamics state. visible[i, k] is True iff k is an own-side index other than
i itself — self and opposing-side entries are masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.belief._common import _truth_arrays_for_side
from orbitalgym.observations.types import Observation
from orbitalgym.registry import ObservationFnKey, register


@register(ObservationFnKey.TEAMMATE_EPHEMERIS)
@dataclass(frozen=True)
class TeammateEphemerisObservation:
    """One channel: full-state measurement of own-side teammates, self excluded.

    Layout requirements: `n_guards`, `n_bandits`, `dynamics_state_dim`.
    """

    layout: Any
    sigma: float = 5.0

    def __call__(self, env_state, actions, side, params, key, t):
        del actions, params, t  # default impls ignore actions
        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        n_self = own_truth.shape[0]
        n_tgt = opp_truth.shape[0]
        n_total = n_self + n_tgt
        d = self.layout.dynamics_state_dim
        dtype = own_truth.dtype

        stacked = jnp.concatenate([own_truth, opp_truth], axis=0)  # (n_total, d)
        target_states = jnp.broadcast_to(stacked[None, :, :], (n_self, n_total, d))
        if self.sigma > 0:
            noise = jnp.asarray(self.sigma, dtype=dtype) * jax.random.normal(
                key, (n_self, n_total, d), dtype=dtype
            )
        else:
            noise = jnp.zeros((n_self, n_total, d), dtype=dtype)
        obs = target_states + noise

        own_idx = jnp.arange(n_self)
        col_idx = jnp.arange(n_total)
        # True iff k < n_self (own side) and k != i (excludes self).
        is_own_side = col_idx[None, :] < n_self
        is_not_self = col_idx[None, :] != own_idx[:, None]
        visible = jnp.logical_and(is_own_side, is_not_self)

        H = jnp.eye(d, dtype=dtype)  # noqa: N806
        sigma_sq = self.sigma**2 if self.sigma > 0 else 1e-12
        R = jnp.eye(d, dtype=dtype) * jnp.asarray(sigma_sq, dtype=dtype)  # noqa: N806

        return (
            Observation(
                obs=obs,
                visible=visible,
                obs_matrix=H,
                obs_noise=R,
                visibility_score_fn=None,
            ),
        )
