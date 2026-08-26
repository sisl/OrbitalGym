"""ConicalObservation — body-frame-cone-gated full-state measurements.

Per (observer, target) pair, visibility is True iff the target's relative
position vector lies inside ANY of the observer's body-fixed cones, after
rotation into the world frame by the observer's quaternion. When visible,
the measurement is the target's full dynamics state plus additive Gaussian
noise; H = I_d, R = sigma^2 * I_d.

Multi-sensor agents are handled internally via a vmap over the sensor axis
— a single Observation channel is returned regardless of sensor count, so
the existing belief updaters consume it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.belief._common import _truth_arrays_for_side
from orbitalgym.observations.types import Observation
from orbitalgym.registry import ObservationFnKey, register


def _quat_wxyz_to_rotation_matrix_jax(q: jax.Array) -> jax.Array:
    """Convert (w, x, y, z) quaternion to a 3x3 body-to-reference rotation matrix.

    JAX-native version (fully traceable, safe to vmap). Normalises the quaternion
    first to guard against near-zero inputs — returns identity for zero quaternion.
    Matches the convention in ``viz.glyphs.quat_to_rotation_matrix``.
    """
    dtype = q.dtype
    norm = jnp.linalg.norm(q)
    q = q / jnp.maximum(norm, jnp.asarray(1e-12, dtype=dtype))
    w, x, y, z = q[0], q[1], q[2], q[3]
    one = jnp.asarray(1.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    return jnp.stack(
        [
            one - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            one - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            one - two * (x * x + y * y),
        ]
    ).reshape(3, 3)


@register(ObservationFnKey.CONICAL)
@dataclass(frozen=True)
class ConicalObservation:
    """One channel: per-pair full-state measurement gated by body-frame cone.

    Parameters
    ----------
    layout:
        Scenario layout object exposing ``n_guards``, ``n_bandits``, and
        ``dynamics_state_dim``.
    sensor_boresights_body:
        Shape ``(k, 3)`` array of unit vectors in the observer's body frame
        defining sensor pointing directions.
    half_angle_rad:
        Half-angle (radians) of each cone — either a scalar (same for all
        sensors) or a ``(k,)`` array / tuple of per-sensor half-angles.
    sigma:
        Standard deviation of additive Gaussian measurement noise. Set to 0.0
        for noiseless (ground-truth) measurements.
    """

    layout: Any
    sensor_boresights_body: jax.Array  # (k, 3) unit vectors
    half_angle_rad: float | jax.Array | tuple[float, ...]  # scalar or (k,)
    sigma: float = (
        1.0  # std on full-state measurement; sigma=0 produces R = 1e-12 * I (near-noiseless)
    )

    def __call__(self, env_state, actions, side, params, key, t):
        del actions, params, t

        own_truth, opp_truth = _truth_arrays_for_side(env_state, side.value)
        n_self = own_truth.shape[0]
        n_tgt = opp_truth.shape[0]
        n_total = n_self + n_tgt
        d = self.layout.dynamics_state_dim
        m = d  # full-state measurement
        pos_dim = 2 if d == 4 else 3

        # Derive dtype from input so the returned Observation is consistent when
        # running with float64 state (e.g. x64 precision mode).
        dtype = own_truth.dtype

        stacked = jnp.concatenate([own_truth, opp_truth], axis=0)  # (n_total, d)
        positions = stacked[:, :pos_dim]  # (n_total, pos_dim)
        if pos_dim == 2:
            # Pad to 3D for cone math.
            positions = jnp.concatenate(
                [positions, jnp.zeros((n_total, 1), dtype=positions.dtype)],
                axis=1,
            )
        observer_pos = positions[:n_self]  # (n_self, 3)

        # Read own quaternions and rotate body-fixed boresights into world frame.
        own_quat = self._read_own_quat(env_state, side)  # (n_self, 4)
        rot = jax.vmap(_quat_wxyz_to_rotation_matrix_jax)(own_quat)  # (n_self, 3, 3)
        # b_world[i, s, :] = rot[i] @ b_body[s]  (n_self, k, 3)
        b_world = jnp.einsum("nij,sj->nsi", rot, self.sensor_boresights_body)
        b_world = b_world / jnp.linalg.norm(b_world, axis=-1, keepdims=True)

        # LOS to every entity (including self — masked out below).
        diff = positions[None, :, :] - observer_pos[:, None, :]  # (n_self, n_total, 3)
        dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        los = diff / jnp.maximum(dist, jnp.asarray(1e-12, dtype=dtype))  # safe normalise

        # Per (observer, sensor, target) cosine. Shape (n_self, k, n_total).
        cos_theta = jnp.einsum("nsi,nti->nst", b_world, los)
        # Resolve per-sensor half-angles to shape (k,).
        if isinstance(self.half_angle_rad, tuple):
            half_angles = jnp.asarray(self.half_angle_rad)
        else:
            half_angles = jnp.broadcast_to(
                jnp.asarray(self.half_angle_rad),
                (self.sensor_boresights_body.shape[0],),
            )
        cos_thresh = jnp.cos(half_angles)  # (k,)
        in_cone = cos_theta >= cos_thresh[None, :, None]  # (n_self, k, n_total)
        any_sensor = jnp.any(in_cone, axis=1)  # (n_self, n_total)

        # AND with opposing-side mask (excludes self pairs and same-side entities).
        opposing_mask = jnp.arange(n_total) >= n_self  # (n_total,)
        opposing_mask = jnp.broadcast_to(opposing_mask[None, :], (n_self, n_total))
        visible = jnp.logical_and(any_sensor, opposing_mask)

        # Measurement: full target state + Gaussian noise.
        target_states = jnp.broadcast_to(stacked[None, :, :], (n_self, n_total, d))
        if self.sigma > 0:
            noise = jnp.asarray(self.sigma, dtype=dtype) * jax.random.normal(
                key, (n_self, n_total, m), dtype=dtype
            )
        else:
            noise = jnp.zeros((n_self, n_total, m), dtype=dtype)
        obs = target_states + noise

        H = jnp.eye(m, dtype=dtype)  # noqa: N806
        # sigma=0 produces R = 1e-12 * I (near-noiseless) rather than I so that
        # Kalman filter updates remain numerically valid and correctly weight
        # near-perfect measurements instead of treating them as unit-noise.
        sigma_sq = self.sigma**2 if self.sigma > 0 else 1e-12
        R_mat = jnp.eye(m, dtype=dtype) * jnp.asarray(sigma_sq, dtype=dtype)  # noqa: N806

        # Closure for negative-information updates: signed-distance score
        # in radians to the *best-margin* sensor cone. Closes over the
        # rotated boresights, observer positions, and per-sensor half-angles.
        boresights_world = b_world  # (n_self, k, 3)
        observer_pos_3d = observer_pos  # (n_self, 3) — already padded to 3D above

        def visibility_score_fn(particles: jax.Array) -> jax.Array:
            # particles: (N_obs, N_total, K, d)
            particle_pos = particles[..., :pos_dim]
            if pos_dim == 2:
                pad_shape = particle_pos.shape[:-1] + (1,)
                particle_pos = jnp.concatenate(
                    [particle_pos, jnp.zeros(pad_shape, dtype=particle_pos.dtype)], axis=-1
                )
            diff = particle_pos - observer_pos_3d[:, None, None, :]  # (N_obs, N_total, K, 3)
            dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
            los = diff / jnp.maximum(dist, jnp.asarray(1e-12, dtype=dtype))
            cos_theta = jnp.einsum("osi,otki->ostk", boresights_world, los)
            angle = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
            score_per_sensor = half_angles[None, :, None, None] - angle
            return jnp.max(score_per_sensor, axis=1)

        return (
            Observation(
                obs=obs,
                visible=visible,
                obs_matrix=H,
                obs_noise=R_mat,
                visibility_score_fn=visibility_score_fn,
            ),
        )

    @staticmethod
    def _read_own_quat(env_state, side):
        from orbitalgym.env.types import Side

        if side is Side.GUARD:
            return env_state.guards.quat
        return env_state.bandits.quat
