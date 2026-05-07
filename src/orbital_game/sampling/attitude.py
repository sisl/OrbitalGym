"""Attitude (quaternion + body-rate) IC samplers.

Each sampler is a frozen dataclass with __call__(n_vehicles, key) -> (quat, omega):
- quat:  (n, 4) wxyz, unit-norm
- omega: (n, 3) body-frame rates (rad/s)

Composes with translational samplers (RelativeKeplerian, RelativeEllipse) via
their `attitude_sampler` field. Backwards-compatible: when no sampler is
configured, IC defaults to identity quaternion + zero rates (the existing
behavior of `Attitude.zeros()` / `BodyRates.zeros()`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.registry import AttitudeSamplerKey, register


@register(AttitudeSamplerKey.IDENTITY)
@dataclass(frozen=True)
class IdentityAttitude:
    """Identity quaternion (1, 0, 0, 0); zero body rates. Default behavior."""

    def __call__(self, n_vehicles: int, key: jax.Array):
        del key
        quat = jnp.zeros((n_vehicles, 4)).at[:, 0].set(1.0)
        omega = jnp.zeros((n_vehicles, 3))
        return quat, omega


@register(AttitudeSamplerKey.FIXED)
@dataclass(frozen=True)
class FixedAttitude:
    """Pinned quaternion + body rates. Both broadcast to all N vehicles.

    Useful for demonstrations and deterministic test scenarios where the user
    wants every spacecraft to start with the same known orientation and slow
    spin.
    """

    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    omega_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __call__(self, n_vehicles: int, key: jax.Array):
        del key
        q = jnp.asarray(self.quat_wxyz)
        q = q / jnp.linalg.norm(q)  # normalize for safety
        quat = jnp.broadcast_to(q, (n_vehicles, 4))
        omega = jnp.broadcast_to(jnp.asarray(self.omega_rad_s), (n_vehicles, 3))
        return quat, omega


@register(AttitudeSamplerKey.UNIFORM_QUAT)
@dataclass(frozen=True)
class UniformAttitude:
    """Uniformly random quaternion (Marsaglia method); zero body rates.

    Samples q from N(0, I_4)^N and normalizes. Canonicalizes to the
    hemisphere with w >= 0 (since q and -q represent the same rotation).
    """

    def __call__(self, n_vehicles: int, key: jax.Array):
        q = jax.random.normal(key, (n_vehicles, 4))
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        # Canonicalize: flip sign if w < 0 so quaternions are uniquely represented.
        sign = jnp.where(q[:, 0:1] < 0, -1.0, 1.0)
        q = q * sign
        omega = jnp.zeros((n_vehicles, 3))
        return q, omega


@register(AttitudeSamplerKey.UNIFORM_RATES)
@dataclass(frozen=True)
class UniformBodyRates:
    """Identity quaternion; uniform random body rates per axis.

    omega ~ Uniform(-omega_max, omega_max) per axis, per vehicle.
    """

    omega_max_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __call__(self, n_vehicles: int, key: jax.Array):
        omega_max = jnp.asarray(self.omega_max_rad_s)
        omega = jax.random.uniform(key, (n_vehicles, 3), minval=-omega_max, maxval=omega_max)
        quat = jnp.zeros((n_vehicles, 4)).at[:, 0].set(1.0)
        return quat, omega


@register(AttitudeSamplerKey.UNIFORM_QUAT_AND_RATES)
@dataclass(frozen=True)
class UniformAttitudeAndRates:
    """Both quaternion and body rates randomized."""

    omega_max_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __call__(self, n_vehicles: int, key: jax.Array):
        k_q, k_o = jax.random.split(key, 2)
        # Random quaternion via Marsaglia.
        q = jax.random.normal(k_q, (n_vehicles, 4))
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        sign = jnp.where(q[:, 0:1] < 0, -1.0, 1.0)
        q = q * sign
        # Random body rates.
        omega_max = jnp.asarray(self.omega_max_rad_s)
        omega = jax.random.uniform(k_o, (n_vehicles, 3), minval=-omega_max, maxval=omega_max)
        return q, omega
