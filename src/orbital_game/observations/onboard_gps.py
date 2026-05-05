"""OnboardGPSObservation — high-accuracy self-state measurement.

Each observer i gets a noisy reading of its own dynamics state. visible[i, k]
is True iff k == i (own self) — all other entries masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbital_game.belief._common import _truth_arrays_for_side
from orbital_game.observations.types import Observation
from orbital_game.registry import ObservationFnKey, register


@register(ObservationFnKey.ONBOARD_GPS)
@dataclass(frozen=True)
class OnboardGPSObservation:
    """One channel: full self-state with isotropic Gaussian noise.

    Layout requirements: `n_guards`, `n_bandits`, `dynamics_state_dim`.
    """

    layout: Any
    sigma_gps: float = 0.1

    def __call__(self, env_state, actions, side, params, key, t):
        del actions, params, t  # default impls ignore actions
        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        n_self = own_truth.shape[0]
        n_tgt = opp_truth.shape[0]
        n_total = n_self + n_tgt
        d = self.layout.dynamics_state_dim

        noise = self.sigma_gps * jax.random.normal(key, (n_self, n_total, d))
        self_truth = jnp.zeros((n_self, n_total, d))
        self_truth = self_truth.at[jnp.arange(n_self), jnp.arange(n_self), :].set(own_truth)
        obs = self_truth + noise
        visible = jnp.eye(n_self, n_total, dtype=bool)
        H = jnp.eye(d)  # noqa: N806
        R = jnp.eye(d) * self.sigma_gps**2  # noqa: N806
        return (Observation(obs=obs, visible=visible, obs_matrix=H, obs_noise=R),)
