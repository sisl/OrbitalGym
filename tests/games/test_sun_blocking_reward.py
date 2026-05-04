"""Unit tests for the KSP-DG-aligned `sun_blocking_kernel`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.games.sun_blocking import sun_blocking_kernel


def test_kernel_peak_at_collinear_in_front():
    """Bandit between sun and guard at d_target → reward ≈ +1."""
    sun_eci = jnp.array([1.5e11, 0.0, 0.0])  # 1 AU
    guard_eci = jnp.array([0.0, 0.0, 0.0])
    # Bandit on the sun-side of guard at exactly d_target.
    d_target = 500.0
    bandit_eci = jnp.array([d_target, 0.0, 0.0])
    r = float(
        sun_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=bandit_eci,
            sun_eci=sun_eci,
            target_viewing_distance_m=d_target,
            range_decay_coef=4e-6,
        )
    )
    assert abs(r - 1.0) < 1e-5


def test_kernel_trough_at_anti_collinear_in_front():
    """Guard between sun and bandit at d_target → reward ≈ -1."""
    sun_eci = jnp.array([1.5e11, 0.0, 0.0])
    guard_eci = jnp.array([0.0, 0.0, 0.0])
    d_target = 500.0
    # Bandit on the far-from-sun side: bandit-to-guard points toward sun,
    # bandit-to-sun also points toward sun ⇒ dot product ≈ +1, -dot ≈ -1.
    bandit_eci = jnp.array([-d_target, 0.0, 0.0])
    r = float(
        sun_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=bandit_eci,
            sun_eci=sun_eci,
            target_viewing_distance_m=d_target,
            range_decay_coef=4e-6,
        )
    )
    assert abs(r - (-1.0)) < 1e-5


def test_kernel_decays_with_range():
    """At θ=π but d=10·d_target, reward ≈ 0 (range factor crushes it)."""
    sun_eci = jnp.array([1.5e11, 0.0, 0.0])
    guard_eci = jnp.array([0.0, 0.0, 0.0])
    d_target = 500.0
    decay = 4e-6
    bandit_eci = jnp.array([10.0 * d_target, 0.0, 0.0])  # collinear, far
    r = float(
        sun_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=bandit_eci,
            sun_eci=sun_eci,
            target_viewing_distance_m=d_target,
            range_decay_coef=decay,
        )
    )
    assert abs(r) < 1e-3


def test_kernel_zero_perpendicular():
    """Bandit perpendicular to sun line at d_target ⇒ angular factor ≈ 0."""
    sun_eci = jnp.array([1.5e11, 0.0, 0.0])
    guard_eci = jnp.array([0.0, 0.0, 0.0])
    d_target = 500.0
    bandit_eci = jnp.array([0.0, d_target, 0.0])  # perpendicular to sun line
    r = float(
        sun_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=bandit_eci,
            sun_eci=sun_eci,
            target_viewing_distance_m=d_target,
            range_decay_coef=4e-6,
        )
    )
    assert abs(r) < 1e-5


def test_kernel_vmap_compatible():
    """Kernel runs under vmap over a batch of bandit positions."""
    sun_eci = jnp.array([1.5e11, 0.0, 0.0])
    guard_eci = jnp.array([0.0, 0.0, 0.0])
    d_target = 500.0
    bandits = jnp.stack(
        [
            jnp.array([+d_target, 0.0, 0.0]),
            jnp.array([-d_target, 0.0, 0.0]),
            jnp.array([0.0, d_target, 0.0]),
        ]
    )
    rewards = jax.vmap(
        lambda b: sun_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=b,
            sun_eci=sun_eci,
            target_viewing_distance_m=d_target,
            range_decay_coef=4e-6,
        )
    )(bandits)
    assert rewards.shape == (3,)
    assert float(rewards[0]) > 0.99
    assert float(rewards[1]) < -0.99
    assert abs(float(rewards[2])) < 1e-5
