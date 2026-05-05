"""RangeLimitedObservation - distance-gated target position measurements.

Each observer measures the position of every opposing-side entity that is
within `sensor_range_m`. Self pairs and out-of-range pairs are masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbital_game.belief._common import _truth_arrays_for_side
from orbital_game.observations.types import Observation
from orbital_game.registry import ObservationFnKey, register


@register(ObservationFnKey.RANGE_LIMITED)
@dataclass(frozen=True)
class RangeLimitedObservation:
    """One channel: per-pair 3-D position measurement gated by distance.

    H selects the first 3 components (position) from the d-dim dynamics state.
    Layout requirements: `n_guards`, `n_bandits`, `dynamics_state_dim`.
    """

    layout: Any
    sensor_range_m: float
    sigma_range: float = 1.0

    def __call__(self, env_state, actions, side, params, key, t):
        del actions, params, t  # default impls ignore actions
        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        n_self = own_truth.shape[0]
        n_tgt = opp_truth.shape[0]
        n_total = n_self + n_tgt
        d = self.layout.dynamics_state_dim
        m = 3

        stacked = jnp.concatenate([own_truth, opp_truth], axis=0)
        positions = stacked[:, :3]
        observer_pos = own_truth[:, :3]
        diff = positions[None, :, :] - observer_pos[:, None, :]
        distance = jnp.linalg.norm(diff, axis=-1)
        opposing_mask = jnp.arange(n_total) >= n_self
        opposing_mask = jnp.broadcast_to(opposing_mask[None, :], (n_self, n_total))
        in_range = distance <= self.sensor_range_m
        visible = jnp.logical_and(opposing_mask, in_range)

        noise = self.sigma_range * jax.random.normal(key, (n_self, n_total, m))
        obs = jnp.broadcast_to(positions[None, :, :], (n_self, n_total, m)) + noise

        H = jnp.eye(m, d)  # noqa: N806
        R = jnp.eye(m) * self.sigma_range**2 if self.sigma_range > 0 else jnp.eye(m)  # noqa: N806
        return (Observation(obs=obs, visible=visible, obs_matrix=H, obs_noise=R),)
