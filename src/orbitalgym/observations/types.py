"""Observation channel payload and flatten helper.

Each ObservationFn returns a tuple of `Observation` channels — one per sensor
modality (GPS, range-limited, etc.). The belief updater consumes the tuple
and sequentially folds each channel's correction.

`flatten_observations` is a deterministic concatenation used by adapters
(Gymnasium, PettingZoo, POMDP) that need to expose obs as a single flat array.
"""

from __future__ import annotations

from collections.abc import Callable

import flax.struct
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class Observation:
    """One sensor channel's measurement payload.

    Shape conventions:
      obs:        (N_obs, N_total, m)         per-pair measurement
      visible:    (N_obs, N_total) bool       per-pair visibility mask
      obs_matrix: (m, d)                       linear measurement matrix H
      obs_noise:  (m, m) or (N_obs, N_total, m, m)  measurement noise R

    `obs_noise` is either shared across every pair, shape (m, m), or
    conditioned on the pair, shape (N_obs, N_total, m, m). Use
    `noise_for(i, j)` to read the (m, m) block for one pair regardless of
    which form a channel supplies.

    `obs_fn` (optional): nonlinear measurement function h(x) -> z. When set,
    the EKF updater computes H = jax.jacfwd(obs_fn)(predicted_mean) per pair
    and uses obs_fn directly for the innovation. The KF updater rejects
    channels with obs_fn set.

    `visibility_score_fn` (optional): closure
    `(particles: (N_obs, N_total, K, d)) -> (N_obs, N_total, K)` returning
    a signed-distance score to the channel's visibility boundary in
    native units — positive inside, negative outside. Consumed by the
    particle filter for negative-information updates on non-detections.
    Non-gated channels (e.g. GPS) leave it None.
    """

    obs: jax.Array
    visible: jax.Array
    obs_matrix: jax.Array
    obs_noise: jax.Array
    obs_fn: Callable | None = flax.struct.field(default=None, pytree_node=False)
    visibility_score_fn: Callable | None = flax.struct.field(default=None, pytree_node=False)

    def noise_for(self, i, j) -> jax.Array:
        """Measurement noise `(m, m)` for the (observer i, entity j) pair.

        Accepts traced indices, so belief updaters can call it from inside
        a `vmap` over the observer and entity axes.
        """
        if self.obs_noise.ndim == 2:
            return self.obs_noise
        return self.obs_noise[i, j]


def flatten_observations(channels: tuple[Observation, ...]) -> jax.Array:
    """Concatenate every channel's `obs` array into a single 1-D vector.

    Used by adapters that publish a flat per-side observation (Gymnasium
    Box space, POMDP per-side observation). The flattened layout is
    deterministic for a fixed observation function.
    """
    if not channels:
        raise ValueError("flatten_observations requires at least one channel")
    return jnp.concatenate([c.obs.reshape(-1) for c in channels])


def flatten_observations_per_agent(channels: tuple[Observation, ...]) -> jax.Array:
    """Flatten per-pair observations to one row per observer.

    Each channel's `obs` has shape `(N_self, N_total, m)`. This helper
    keeps the leading observer axis intact and flattens the trailing pair
    and feature axes, then concatenates channels along the inner axis.
    Returns shape `(N_self, total_per_agent_dim)`. Observer i's view is
    row i — distinct from every other observer's row.

    Used by adapters that need genuine per-agent observations
    (PettingZoo). Use `flatten_observations` instead when the consumer
    treats the side as one unit (Gymnasium, POMDP).
    """
    if not channels:
        raise ValueError("flatten_observations_per_agent requires at least one channel")
    # Each channel: (N_self, N_total, m) → (N_self, N_total * m)
    per_channel = [c.obs.reshape(c.obs.shape[0], -1) for c in channels]
    return jnp.concatenate(per_channel, axis=-1)
