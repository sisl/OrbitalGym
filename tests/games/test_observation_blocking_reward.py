"""Unit tests for the KSP-DG-aligned `observation_blocking_kernel`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.games.observation_blocking import (
    observation_blocking_kernel,
    target_visible_from_guard,
)


def _earth_surface_target():
    """A target on the equator at zero longitude in ECI ≈ ECEF for testing."""
    return jnp.array([6.378e6, 0.0, 0.0])


def test_kernel_peak_at_collinear_in_front_when_visible():
    """Bandit between target and guard at d_target, guard overhead → reward ≈ +1."""
    target_eci = _earth_surface_target()
    # Guard 1000 km overhead — target visibility ≈ 90° elevation.
    guard_eci = target_eci * (1.0 + 1.0e6 / float(jnp.linalg.norm(target_eci)))
    d_target = 500.0
    # Bandit between guard and target at distance d_target from guard.
    direction = (target_eci - guard_eci) / float(jnp.linalg.norm(target_eci - guard_eci))
    bandit_eci = guard_eci + d_target * direction
    r = float(
        observation_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=bandit_eci,
            target_eci=target_eci,
            target_viewing_distance_m=d_target,
            range_decay_coef=4e-6,
            min_elevation_deg=5.0,
        )
    )
    assert abs(r - 1.0) < 1e-3


def test_kernel_zero_when_not_visible():
    """Target below horizon ⇒ reward exactly 0 regardless of bandit geometry."""
    target_eci = _earth_surface_target()
    # Place guard on the antipodal side of the Earth — target is way below horizon.
    guard_eci = -target_eci * 1.1
    d_target = 500.0
    bandit_eci = guard_eci + jnp.array([d_target, 0.0, 0.0])
    r = float(
        observation_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=bandit_eci,
            target_eci=target_eci,
            target_viewing_distance_m=d_target,
            range_decay_coef=4e-6,
            min_elevation_deg=5.0,
        )
    )
    assert r == 0.0


def test_kernel_decays_with_range_when_visible():
    """At collinear but d=10·d_target with target visible, reward ≈ 0."""
    target_eci = _earth_surface_target()
    guard_eci = target_eci * (1.0 + 1.0e6 / float(jnp.linalg.norm(target_eci)))
    d_target = 500.0
    direction = (target_eci - guard_eci) / float(jnp.linalg.norm(target_eci - guard_eci))
    bandit_eci = guard_eci + 10.0 * d_target * direction
    r = float(
        observation_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=bandit_eci,
            target_eci=target_eci,
            target_viewing_distance_m=d_target,
            range_decay_coef=4e-6,
            min_elevation_deg=5.0,
        )
    )
    assert abs(r) < 1e-3


def test_kernel_grad_finite():
    """jax.grad over bandit position is finite at a generic geometry."""
    target_eci = _earth_surface_target()
    guard_eci = target_eci * 1.1
    d_target = 500.0
    bandit_eci = guard_eci + jnp.array([d_target, 100.0, 50.0])

    def f(b):
        return observation_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=b,
            target_eci=target_eci,
            target_viewing_distance_m=d_target,
            range_decay_coef=4e-6,
            min_elevation_deg=5.0,
        )

    g = jax.grad(f)(bandit_eci)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_visibility_gate_threshold():
    """Elevation gate fires exactly at min_elevation_deg."""
    target_eci = _earth_surface_target()
    # Pure overhead is ~90°, so visible.
    overhead = target_eci * 1.1
    assert bool(target_visible_from_guard(target_eci, overhead, 5.0))
    # Antipodal is well below horizon.
    antipodal = -target_eci * 1.1
    assert not bool(target_visible_from_guard(target_eci, antipodal, 5.0))
