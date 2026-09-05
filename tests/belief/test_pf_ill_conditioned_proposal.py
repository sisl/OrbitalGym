"""PF proposal on covariances that span many orders of magnitude.

A ring prior over kilometres of position sits in the same state vector as
metres-per-second of velocity, so the pair's covariance can carry a
condition number near the float32 limit. The proposal has to stay finite
there, and has to keep sampling the covariance it claims to.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp

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
RING_RADIUS_M = 30_000.0
MEAN_MOTION = 1.08e-3
PROCESS_NOISE = jnp.diag(jnp.array([2.27] * 3 + [0.4545] * 3) ** 2)


class _Layout:
    def __init__(self) -> None:
        self.n_guards = 1
        self.n_bandits = 1
        self.dynamics_state_dim = D


def _drift(x: jax.Array, u: jax.Array, dt: float) -> jax.Array:
    velocity = x[3:] + u
    return jnp.concatenate([x[:3] + velocity * dt, velocity])


def _ill_conditioned_covariance() -> jax.Array:
    """Symmetric PSD matrix whose diagonal spans seven orders of magnitude."""
    scale = jnp.array([1.2e3, 2.4e3, 2.3, 1.3, 2.5, 0.45])
    correlation = jnp.full((D, D), 0.97) + 0.03 * jnp.eye(D)
    return (scale[:, None] * scale[None, :]) * correlation


def test_psd_cholesky_factorizes_an_ill_conditioned_covariance():
    a = _ill_conditioned_covariance()
    assert float(jnp.linalg.cond(a)) > 1e6

    factor = _psd_cholesky(a)
    assert bool(jnp.all(jnp.isfinite(factor)))
    reconstructed = factor @ factor.T
    relative = jnp.abs(reconstructed - a) / jnp.sqrt(jnp.outer(jnp.diag(a), jnp.diag(a)))
    assert float(jnp.max(relative)) < 1e-4


def test_psd_solve_matches_a_well_scaled_reference():
    a = _ill_conditioned_covariance()
    b = jnp.eye(D)[:, :2] * jnp.array([1.0e3, 1.0])
    x = _psd_solve(a, b)
    assert bool(jnp.all(jnp.isfinite(x)))
    residual = a @ x - b
    assert float(jnp.max(jnp.abs(residual) / jnp.linalg.norm(b, axis=0))) < 1e-2


def test_proposal_samples_the_conditioned_covariance():
    """The drawn cloud must carry the covariance the importance weight assumes.

    One pair, an identical prior cloud so the pair's prior covariance is
    exactly ``Q``, and a position-only channel: the conditioned covariance
    is then available in closed form to compare against.
    """
    k_particles = 16384
    H = jnp.eye(3, D)  # noqa: N806
    R = jnp.eye(3) * 9.0  # noqa: N806
    channel = Observation(
        obs=jnp.zeros((1, 1, 3)),
        visible=jnp.array([[True]]),
        obs_matrix=H,
        obs_noise=R,
    )
    updater = ParticleFilterBeliefUpdater(dynamics_fn=_drift, process_noise=PROCESS_NOISE, dt=DT)
    propagated = jnp.zeros((1, 1, k_particles, D))
    prior_log_w = jnp.full((1, 1, k_particles), -jnp.log(float(k_particles)))
    q_reg = PROCESS_NOISE + 1e-12 * jnp.eye(D)
    drawn, log_inc = updater._propose(
        propagated,
        prior_log_w,
        (channel,),
        q_reg,
        jnp.linalg.cholesky(q_reg),
        jax.random.PRNGKey(0),
    )
    assert bool(jnp.all(jnp.isfinite(drawn)))
    assert bool(jnp.all(jnp.isfinite(log_inc)))

    gain = q_reg @ H.T @ jnp.linalg.inv(H @ q_reg @ H.T + R)
    spread = jnp.eye(D) - gain @ H
    expected = spread @ q_reg @ spread.T + gain @ R @ gain.T
    empirical = jnp.cov(drawn[0, 0].T)
    tolerance = 0.1 * jnp.sqrt(jnp.outer(jnp.diag(expected), jnp.diag(expected)))
    assert bool(jnp.all(jnp.abs(empirical - expected) < tolerance))


def test_ring_prior_update_stays_finite_at_256_particles():
    """A wide ring opponent pair beside a GPS-pinned self pair.

    That is the pairing the cone scenario starts from: the self pair's
    cloud has effectively no spread while the opponent pair's spans
    kilometres along a curve, so its covariance is both huge and nearly
    rank-deficient. With the opponent in the cone the proposal has to
    condition that covariance on a measurement far sharper than it.
    """
    layout = _Layout()
    n_particles = 256
    guard_truth = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    bandit_truth = jnp.array([[RING_RADIUS_M, 0.0, 0.0, 0.0, 2.0, 0.0]])
    env_state = SimpleNamespace(
        guards=SimpleNamespace(rtn=guard_truth, quat=jnp.array([[1.0, 0.0, 0.0, 0.0]])),
        bandits=SimpleNamespace(rtn=bandit_truth, quat=jnp.array([[1.0, 0.0, 0.0, 0.0]])),
    )
    initializer = ParticleFilterRingInitializer(
        layout=layout,
        ring_radius_m=RING_RADIUS_M,
        mean_motion_rad_s=MEAN_MOTION,
        n_particles=n_particles,
    )
    belief = initializer(env_state, Side.GUARD, jax.random.PRNGKey(0))

    opponent_cloud = belief.particles[0, 1]
    centered = opponent_cloud - jnp.mean(opponent_cloud, axis=0)
    cloud_cov = centered.T @ centered / n_particles + PROCESS_NOISE
    assert float(jnp.linalg.cond(cloud_cov)) > 1e9

    obs_fn = CompositeObservation(
        constituents=(
            OnboardGPSObservation(layout=layout, sigma_gps=2.0),
            ConicalObservation(
                layout=layout,
                sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]]),
                half_angle_rad=0.52,
                sigma_floor=4.0,
                sigma_range_frac=0.0,
            ),
        )
    )
    channels = obs_fn(env_state, None, Side.GUARD, None, jax.random.PRNGKey(1), 0.0)
    assert bool(channels[1].visible[0, 1]), "opponent must be in the cone for this case"

    updater = ParticleFilterBeliefUpdater(
        dynamics_fn=_drift,
        process_noise=PROCESS_NOISE,
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
