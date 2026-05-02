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
      obs_noise:  (m, m)                       measurement noise R

    `obs_fn` (optional): nonlinear measurement function h(x) -> z. When set,
    the EKF updater computes H = jax.jacfwd(obs_fn)(predicted_mean) per pair
    and uses obs_fn directly for the innovation. The KF updater rejects
    channels with obs_fn set.
    """

    obs: jax.Array
    visible: jax.Array
    obs_matrix: jax.Array
    obs_noise: jax.Array
    obs_fn: Callable | None = flax.struct.field(default=None, pytree_node=False)


def flatten_observations(channels: tuple[Observation, ...]) -> jax.Array:
    """Concatenate every channel's `obs` array into a single 1-D vector.

    Used by adapters that publish a flat observation (Gymnasium Box space,
    PettingZoo per-agent obs). The flattened layout is deterministic for a
    fixed observation function.
    """
    if not channels:
        raise ValueError("flatten_observations requires at least one channel")
    return jnp.concatenate([c.obs.reshape(-1) for c in channels])


def coerce_obs_to_flat_array(obs: jax.Array | tuple[Observation, ...]) -> jax.Array:
    """Coerce an obs payload to a flat jax.Array.

    Transitional helper used by adapters during the migration to multi-channel
    observations. After all observation fns return tuple[Observation, ...],
    callers should switch to `flatten_observations` directly.
    """
    if isinstance(obs, tuple):
        return flatten_observations(obs)
    return obs
