"""PF tracking under a composite of a self-state channel and a gated channel.

Covers the two cells a cone-sensor scenario depends on: the observer's own
state, measured every step by onboard GPS, and an opponent that the cone
only sees intermittently.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from orbitalgym.belief.pf import ParticleFilterBelief, ParticleFilterBeliefUpdater
from orbitalgym.env.types import Side
from orbitalgym.observations.composite import CompositeObservation
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.observations.negative_info import Hard, Off
from orbitalgym.observations.onboard_gps import OnboardGPSObservation

D = 6
DT = 10.0
SIGMA_GPS = 2.0
SIGMA_FLOOR = 1.0
SIGMA_RANGE_FRAC = 0.005
HALF_ANGLE_RAD = 0.35
# Process noise sized to a full-authority maneuver per step, as a scenario
# that must follow an unmodelled opponent maneuver has to size it.
Q_VEL = 0.4545
Q_POS = 0.5 * Q_VEL * DT
PROCESS_NOISE = jnp.diag(jnp.array([Q_POS] * 3 + [Q_VEL] * 3) ** 2)

IDENTITY_QUAT = jnp.array([[1.0, 0.0, 0.0, 0.0]])
# 90 degrees about cross-track: swings the +R boresight onto +T.
YAW_QUAT = jnp.array([[jnp.sqrt(0.5), 0.0, 0.0, jnp.sqrt(0.5)]])


class _Layout:
    def __init__(self) -> None:
        self.n_guards = 1
        self.n_bandits = 1
        self.dynamics_state_dim = D


def _drift(x: jax.Array, u: jax.Array, dt: float) -> jax.Array:
    """Double-integrator step: velocity impulse at the start of the interval."""
    velocity = x[3:] + u
    return jnp.concatenate([x[:3] + velocity * dt, velocity])


def _observation_fn(layout: _Layout) -> CompositeObservation:
    return CompositeObservation(
        constituents=(
            OnboardGPSObservation(layout=layout, sigma_gps=SIGMA_GPS),
            ConicalObservation(
                layout=layout,
                sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]]),
                half_angle_rad=HALF_ANGLE_RAD,
                sigma_floor=SIGMA_FLOOR,
                sigma_range_frac=SIGMA_RANGE_FRAC,
            ),
        )
    )


def _env_state(guard: jax.Array, bandit: jax.Array, quat: jax.Array) -> SimpleNamespace:
    """Env state stub carrying one guard and one bandit, both pointed by `quat`."""
    return SimpleNamespace(
        guards=SimpleNamespace(rtn=guard[None, :], quat=quat),
        bandits=SimpleNamespace(rtn=bandit[None, :], quat=quat),
    )


def _belief_from(particles: jax.Array) -> ParticleFilterBelief:
    n_obs, n_total, k, _ = particles.shape
    return ParticleFilterBelief(
        particles=particles,
        log_weights=jnp.full((n_obs, n_total, k), -jnp.log(float(k))),
        n_eff=jnp.full((n_obs, n_total), float(k)),
        weight_entropy=jnp.full((n_obs, n_total), float(jnp.log(k))),
        resampled=jnp.zeros((n_obs, n_total), dtype=bool),
    )


def _updater(negative_info) -> ParticleFilterBeliefUpdater:
    return ParticleFilterBeliefUpdater(
        dynamics_fn=_drift,
        process_noise=PROCESS_NOISE,
        dt=DT,
        n_eff_threshold=0.5,
        resample_jitter_scale=0.05,
        negative_info=negative_info,
    )


@pytest.mark.parametrize("side", [Side.GUARD, Side.BANDIT])
@pytest.mark.parametrize("negative_info", [Off(), Hard()], ids=["off", "hard"])
def test_own_state_stays_pinned_by_onboard_gps(side, negative_info):
    """Onboard GPS must hold the own-state cell whatever the cone reports.

    The opponent sits far along +T, outside the +R cone for the whole run,
    so every step is a non-detection on the gated channel. That must not
    touch the observer's own-state particles.
    """
    layout = _Layout()
    obs_fn = _observation_fn(layout)
    updater = _updater(negative_info)
    key = jax.random.PRNGKey(0)

    own = jnp.array([120.0, -40.0, 15.0, 0.05, -0.12, 0.03])
    opponent = jnp.array([0.0, 4000.0, 0.0, 0.0, 0.0, 0.0])
    k_init, key = jax.random.split(key)
    truth = jnp.stack([own, opponent])
    particles = truth[None, :, None, :] + jax.random.normal(k_init, (1, 2, 256, D))
    belief = _belief_from(particles)

    errors = []
    for _ in range(100):
        key, k_dv, k_obs, k_upd = jax.random.split(key, 4)
        dv = Q_VEL * jax.random.normal(k_dv, (3,))
        own = _drift(own, dv, DT)
        opponent = _drift(opponent, jnp.zeros(3), DT)
        guard, bandit = (own, opponent) if side is Side.GUARD else (opponent, own)
        channels = obs_fn(_env_state(guard, bandit, IDENTITY_QUAT), None, side, None, k_obs, 0.0)
        assert not bool(jnp.any(channels[1].visible)), "opponent must stay out of the cone"
        belief = updater(belief, channels, dv[None, :], side, k_upd)
        errors.append(float(jnp.linalg.norm(belief.mean[0, 0, :3] - own[:3])))

    assert max(errors) < 5.0 * SIGMA_GPS


def test_intermittently_visible_opponent_is_tracked_and_does_not_run_away():
    """A cone detection every third step must pull the opponent cell onto it.

    The cell starts several hundred metres off truth, so the run also covers
    a first detection that lands well outside the initial cloud. After the
    detections stop, the cell must decay gracefully rather than run away.
    """
    layout = _Layout()
    obs_fn = _observation_fn(layout)
    updater = _updater(Hard())
    key = jax.random.PRNGKey(1)

    guard = jnp.zeros(D)
    bandit = jnp.array([1000.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    k_init, key = jax.random.split(key)
    offset = jnp.array([250.0, -150.0, 80.0, 0.0, 0.0, 0.0])
    centers = jnp.stack([guard, bandit + offset])
    spread = jnp.array([100.0] * 3 + [1.0] * 3)
    particles = centers[None, :, None, :] + spread * jax.random.normal(k_init, (1, 2, 256, D))
    belief = _belief_from(particles)

    visible_errors, blackout_errors = [], []
    n_visible_steps, n_blackout_steps = 60, 20
    for t in range(n_visible_steps + n_blackout_steps):
        key, k_obs, k_upd = jax.random.split(key, 3)
        bandit = _drift(bandit, jnp.zeros(3), DT)
        seen = t < n_visible_steps and t % 3 == 0
        quat = IDENTITY_QUAT if seen else YAW_QUAT
        channels = obs_fn(_env_state(guard, bandit, quat), None, Side.GUARD, None, k_obs, float(t))
        assert bool(channels[1].visible[0, 1]) == seen
        belief = updater(belief, channels, jnp.zeros((1, 3)), Side.GUARD, k_upd)
        error = float(jnp.linalg.norm(belief.mean[0, 1, :3] - bandit[:3]))
        sigma = SIGMA_FLOOR + SIGMA_RANGE_FRAC * float(jnp.linalg.norm(bandit[:3]))
        if t >= n_visible_steps:
            blackout_errors.append(error)
        elif seen and t > 6:
            visible_errors.append(error / sigma)

    assert max(visible_errors) < 5.0
    assert max(blackout_errors) < 250.0
