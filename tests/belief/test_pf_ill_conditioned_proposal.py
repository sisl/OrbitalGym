"""PF proposal on covariances that span many orders of magnitude.

A ring prior over kilometres of position sits in the same state vector as
metres-per-second of velocity, and the ring is a curve, so a pair's cloud
covariance is both enormous and nearly rank-deficient. Conditioning it on
a measurement has to stay finite, and has to keep sampling the covariance
the importance weight assumes.

The whole test session runs in float64 (see ``tests/conftest.py``), which
hides every one of these failures: they are float32 effects, and float32
is what the package defaults to and what these scenarios run under on
GPU. The ``float32_precision`` fixture drops precision for the duration
of a test, and each case builds its own arrays inside the test — a
module-level literal would be made at import time, in float64, and would
silently promote the whole computation.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

import orbitalgym
from orbitalgym.belief.pf import (
    ParticleFilterBeliefUpdater,
    ParticleFilterRingInitializer,
    _psd_cholesky,
    _psd_solve,
)
from orbitalgym.env.types import Side
from orbitalgym.observations.composite import CompositeObservation
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.observations.negative_info import Hard
from orbitalgym.observations.onboard_gps import OnboardGPSObservation
from orbitalgym.observations.types import Observation

D = 6
DT = 10.0
RING_RADIUS_M = 100_000.0
MEAN_MOTION = 1.08e-3
SENSOR_SIGMA_M = 1.0
PROCESS_NOISE_DIAG = (2.27, 2.27, 2.27, 0.4545, 0.4545, 0.4545)


class _Layout:
    def __init__(self) -> None:
        self.n_guards = 1
        self.n_bandits = 1
        self.dynamics_state_dim = D


@pytest.fixture
def float32_precision():
    """Run one test in the package's default float32, restoring float64 after."""
    orbitalgym.set_precision(jnp.float32)
    try:
        yield
    finally:
        orbitalgym.set_precision(jnp.float64)


def _process_noise() -> jax.Array:
    """Process noise in whatever precision is active."""
    return jnp.diag(jnp.asarray(PROCESS_NOISE_DIAG) ** 2)


def _drift(x: jax.Array, u: jax.Array, dt: float) -> jax.Array:
    velocity = x[3:] + u
    return jnp.concatenate([x[:3] + velocity * dt, velocity])


def _ring_cloud(radius_m: float, k_particles: int = 256) -> jax.Array:
    """Particles evenly spaced along a 2:1 natural-motion ring."""
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, k_particles)
    zeros = jnp.zeros_like(phase)
    return jnp.stack(
        [
            -radius_m * jnp.cos(phase),
            2.0 * radius_m * jnp.sin(phase),
            zeros,
            MEAN_MOTION * radius_m * jnp.sin(phase),
            2.0 * MEAN_MOTION * radius_m * jnp.cos(phase),
            zeros,
        ],
        axis=-1,
    )


def _observation_fn(layout: _Layout) -> CompositeObservation:
    return CompositeObservation(
        constituents=(
            OnboardGPSObservation(layout=layout, sigma_gps=2.0),
            ConicalObservation(
                layout=layout,
                sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]]),
                half_angle_rad=0.52,
                sigma_floor=SENSOR_SIGMA_M,
                sigma_range_frac=0.0,
            ),
        )
    )


def test_psd_cholesky_handles_a_rank_deficient_ring_covariance(float32_precision):
    """A ring cloud's covariance has rank 2 in a 6-dimensional state.

    Factorizing it as it stands cannot work — the null directions round to
    either side of zero — so the helper factorizes the unit-diagonal form
    under a relative floor and rescales.
    """
    cloud = _ring_cloud(3.0e4)
    centered = cloud - jnp.mean(cloud, axis=0)
    cov = centered.T @ centered / cloud.shape[0]
    assert cov.dtype == jnp.float32
    assert int(jnp.linalg.matrix_rank(cov)) < D

    factor = _psd_cholesky(cov)
    assert factor.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(factor)))

    scale = jnp.sqrt(jnp.outer(jnp.diag(cov), jnp.diag(cov)))
    relative = jnp.abs(factor @ factor.T - cov) / jnp.clip(scale, 1e-30)
    assert float(jnp.max(relative)) < 1e-4


def test_psd_solve_is_accurate_across_a_wide_diagonal(float32_precision):
    """Solving against a covariance whose scales span six orders of magnitude."""
    scale = jnp.asarray([1.2e3, 2.4e3, 2.3, 1.3, 2.5, 0.45])
    correlation = jnp.full((D, D), 0.97) + 0.03 * jnp.eye(D)
    a = (scale[:, None] * scale[None, :]) * correlation
    assert a.dtype == jnp.float32
    assert float(jnp.linalg.cond(a)) > 1e6

    b = jnp.eye(D)[:, :2] * jnp.asarray([1.0e3, 1.0])
    x = _psd_solve(a, b)
    assert bool(jnp.all(jnp.isfinite(x)))
    residual = a @ x - b
    assert float(jnp.max(jnp.abs(residual) / jnp.linalg.norm(b, axis=0))) < 1e-2


def _bandwidth_sq(k_particles: int) -> float:
    """Silverman's bandwidth for ``k_particles`` samples in ``D`` dimensions, squared."""
    return (4.0 / (k_particles * (D + 2))) ** (2.0 / (D + 4))


@pytest.mark.parametrize("prior", ["point", "ring"], ids=["point-cloud", "ring-cloud"])
def test_proposal_noise_carries_the_conditioned_covariance(float32_precision, prior):
    """The drawn cloud must carry the covariance the importance weight assumes.

    Compared against the Joseph form of the same conditioning, which stays
    accurate in float32 where ``P - gain H P`` does not: with a ring prior
    the latter comes back with one variance several times too large and
    another driven to zero, or fails to factorize at all.
    """
    k_particles = 8192
    cloud = jnp.zeros((k_particles, D)) if prior == "point" else _ring_cloud(3.0e4, k_particles)
    propagated = cloud[None, None]
    prior_log_w = jnp.full((1, 1, k_particles), -jnp.log(float(k_particles)))

    H = jnp.eye(D)  # noqa: N806
    R = jnp.eye(D)  # noqa: N806
    channel = Observation(
        obs=jnp.zeros((1, 1, D)),
        visible=jnp.array([[True]]),
        obs_matrix=H,
        obs_noise=R,
    )
    q_reg = _process_noise() + 1e-12 * jnp.eye(D)
    updater = ParticleFilterBeliefUpdater(dynamics_fn=_drift, process_noise=_process_noise(), dt=DT)
    drawn, log_inc = updater._propose(
        propagated,
        prior_log_w,
        (channel,),
        q_reg,
        jnp.linalg.cholesky(q_reg),
        jax.random.PRNGKey(0),
    )
    assert drawn.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(drawn)))
    assert bool(jnp.all(jnp.isfinite(log_inc)))

    centered = cloud - jnp.mean(cloud, axis=0)
    cloud_cov = centered.T @ centered / k_particles
    P = _bandwidth_sq(k_particles) * cloud_cov + q_reg  # noqa: N806
    gain = jnp.linalg.solve(H @ P @ H.T + R, H @ P).T
    spread = jnp.eye(D) - gain @ H
    expected = spread @ P @ spread.T + gain @ R @ gain.T

    # Each particle is shifted by its own residual, so isolate the noise.
    shifted = cloud + (jnp.zeros((k_particles, D)) - cloud @ H.T) @ gain.T
    empirical = jnp.cov((drawn[0, 0] - shifted).T)
    ratio = jnp.diag(empirical) / jnp.diag(expected)
    assert float(jnp.min(ratio)) > 0.8
    assert float(jnp.max(ratio)) < 1.25


def test_ring_prior_update_stays_finite_at_256_particles(float32_precision):
    """A wide ring opponent pair beside a GPS-pinned self pair.

    That is the pairing the cone scenario starts from: the self pair's
    cloud has effectively no spread while the opponent pair's spans a ring
    of kilometres, so its covariance is huge and nearly rank-deficient at
    once. With the opponent in the cone, the proposal has to condition
    that covariance on a measurement far sharper than it.
    """
    layout = _Layout()
    n_particles = 256
    quat = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    env_state = SimpleNamespace(
        guards=SimpleNamespace(rtn=jnp.zeros((1, D)), quat=quat),
        bandits=SimpleNamespace(
            rtn=jnp.array([[RING_RADIUS_M, 0.0, 0.0, 0.0, 2.0, 0.0]]), quat=quat
        ),
    )
    initializer = ParticleFilterRingInitializer(
        layout=layout,
        ring_radius_m=RING_RADIUS_M,
        mean_motion_rad_s=MEAN_MOTION,
        n_particles=n_particles,
    )
    belief = initializer(env_state, Side.GUARD, jax.random.PRNGKey(0))
    assert belief.particles.dtype == jnp.float32

    opponent_cloud = belief.particles[0, 1]
    centered = opponent_cloud - jnp.mean(opponent_cloud, axis=0)
    cloud_cov = centered.T @ centered / n_particles + _process_noise()
    assert float(jnp.linalg.cond(cloud_cov)) > 1e10

    channels = _observation_fn(layout)(
        env_state, None, Side.GUARD, None, jax.random.PRNGKey(1), 0.0
    )
    assert bool(channels[1].visible[0, 1]), "opponent must be in the cone for this case"

    updater = ParticleFilterBeliefUpdater(
        dynamics_fn=_drift,
        process_noise=_process_noise(),
        dt=DT,
        n_eff_threshold=0.5,
        resample_jitter_scale=0.05,
        negative_info=Hard(),
    )
    for _ in range(3):
        belief = updater(belief, channels, jnp.zeros((1, 3)), Side.GUARD, jax.random.PRNGKey(2))
        assert bool(jnp.all(jnp.isfinite(belief.particles)))
        assert bool(jnp.all(jnp.isfinite(belief.log_weights)))
        assert bool(jnp.all(jnp.isfinite(belief.mean)))
