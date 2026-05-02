"""Reference observation: full per-pair channel with all-visible mask, no noise.

Replaces the old flat-ground-truth oracle. Returns one Observation channel
where every (observer, tracked-entity) pair is visible and the measurement is
the truth dynamics state of the tracked entity (H = identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from orbital_game.belief._common import _truth_arrays_for_side
from orbital_game.observations.types import Observation
from orbital_game.registry import ObservationFnKey, register


@register(ObservationFnKey.FULL)
@dataclass(frozen=True)
class FullObservation:
    """One channel covering every tracked entity, no noise, always visible.

    Layout requirements: must expose `n_guards`, `n_bandits`,
    `dynamics_state_dim` attributes.
    """

    layout: Any
    epsilon: float = 1e-6

    def __call__(self, env_state, side, params, key, t):
        del params, key, t
        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        n_self = own_truth.shape[0]
        n_tgt = opp_truth.shape[0]
        n_total = n_self + n_tgt
        d = self.layout.dynamics_state_dim
        stacked = jnp.concatenate([own_truth, opp_truth], axis=0)
        obs = jnp.broadcast_to(stacked[None, :, :], (n_self, n_total, d))
        visible = jnp.ones((n_self, n_total), dtype=bool)
        H = jnp.eye(d)  # noqa: N806
        R = jnp.eye(d) * self.epsilon  # noqa: N806
        return (Observation(obs=obs, visible=visible, obs_matrix=H, obs_noise=R),)
