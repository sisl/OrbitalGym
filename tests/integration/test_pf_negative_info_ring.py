"""Integration test: PF negative-info modes monotonically push belief
mass outside the sensor gate over multiple non-detection steps.

Setup: 1 guard at the origin, 1 unobserved target. Guard's belief about
the target is a uniform-in-2D-strip prior of K particles spanning a
strip far longer along x than the sensor range. The strip is positioned
so a substantial fraction of particles starts inside the gate. We drive
the PF with a sequence of non-detection observations (visible=False) and
measure the posterior mass remaining inside the gate after T steps.

Expected ordering at the final step (mass remaining inside the gate):
  inside_mass(Off) >= inside_mass(Soft) >= inside_mass(Hard)

Off leaves all particles untouched (inside_mass == prior fraction).
Soft crushes inside particles proportionally to softness over T steps.
Hard crushes them all at once.

Variance is not a clean monotonic indicator of belief sharpening for
this geometry — removing inside mass can leave a wider remainder. We
therefore use the inside-gate mass as the direct measure of
negative-information weighting strength.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbitalgym.belief.pf import (
    ParticleFilterBelief,
    ParticleFilterBeliefUpdater,
)
from orbitalgym.env.types import Side
from orbitalgym.observations.negative_info import Hard, Off, Soft
from orbitalgym.observations.types import Observation


def _identity_dynamics(x, u, dt):
    del u, dt
    return x


def _strip_particles(
    n_particles: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    key: jax.Array,
) -> jax.Array:
    """K particles uniformly inside an (x, y) box. Returns shape
    (1, 2, K, 4) for 1 observer, 2 targets (self + opp), d=4."""
    k_x, k_y = jax.random.split(key, 2)
    xs = jax.random.uniform(k_x, (n_particles,), minval=x_range[0], maxval=x_range[1])
    ys = jax.random.uniform(k_y, (n_particles,), minval=y_range[0], maxval=y_range[1])
    target_particles = jnp.stack(
        [xs, ys, jnp.zeros_like(xs), jnp.zeros_like(xs)], axis=-1
    )  # (K, 4)
    self_particles = jnp.zeros((n_particles, 4))
    return jnp.stack([self_particles, target_particles], axis=0)[None, :, :, :]  # (1, 2, K, 4)


def _make_belief(particles: jax.Array) -> ParticleFilterBelief:
    n_obs, n_total, k, _ = particles.shape
    log_w = jnp.full((n_obs, n_total, k), -jnp.log(float(k)))
    n_eff = jnp.full((n_obs, n_total), float(k))
    weight_entropy = jnp.full((n_obs, n_total), float(jnp.log(k)))
    resampled = jnp.zeros((n_obs, n_total), dtype=bool)
    return ParticleFilterBelief(
        particles=particles,
        log_weights=log_w,
        n_eff=n_eff,
        weight_entropy=weight_entropy,
        resampled=resampled,
    )


def _make_channel(sensor_range_m: float, observer_xy: tuple[float, float]) -> Observation:
    """Range-limited-style channel with target non-visible. Closure
    captures observer position and computes score = sensor_range_m - dist."""
    observer_pos = jnp.array([[observer_xy[0], observer_xy[1], 0.0]])  # (1, 3)

    def visibility_score_fn(particles):
        # particles: (1, 2, K, 4). Use the first 2 dims as (x, y) and pad z=0.
        particle_pos = jnp.concatenate(
            [particles[..., :2], jnp.zeros(particles.shape[:-1] + (1,))], axis=-1
        )
        diff = particle_pos - observer_pos[:, None, None, :]
        distance = jnp.linalg.norm(diff, axis=-1)
        return sensor_range_m - distance

    # All pairs non-visible (target undetected).
    return Observation(
        obs=jnp.zeros((1, 2, 4)),
        visible=jnp.array([[False, False]]),
        obs_matrix=jnp.eye(4),
        obs_noise=jnp.eye(4),
        visibility_score_fn=visibility_score_fn,
    )


def _inside_gate_mass(
    belief: ParticleFilterBelief,
    sensor_range_m: float,
    observer_xy: tuple[float, float],
) -> float:
    """Total weight of particles that lie inside the sensor gate
    (observer 0, target 1). Pre-detection this equals the fraction of
    particles inside the disk; negative-info weighting should drive it
    toward zero."""
    particles_xy = belief.particles[0, 1, :, :2]  # (K, 2)
    obs_xy = jnp.asarray(observer_xy)
    distances = jnp.linalg.norm(particles_xy - obs_xy[None, :], axis=-1)  # (K,)
    inside = distances < sensor_range_m  # (K,) bool
    weights = jax.nn.softmax(belief.log_weights[0, 1])  # (K,)
    return float(jnp.sum(jnp.where(inside, weights, 0.0)))


def _run_with_mode(
    negative_info,
    n_steps: int = 10,
) -> tuple[float, float]:
    sensor_range_m = 50.0
    # Strip extends well past the sensor range along +x; observer sits
    # at the origin, so a clean fraction of particles starts inside the
    # gate and the negative-info update can drive that fraction toward
    # zero on repeated non-detections.
    x_range = (0.0, 300.0)
    y_range = (-25.0, 25.0)
    observer_xy = (0.0, 0.0)
    n_particles = 400

    key = jax.random.PRNGKey(42)
    particles = _strip_particles(n_particles, x_range, y_range, key)
    belief = _make_belief(particles)
    initial_inside_mass = _inside_gate_mass(belief, sensor_range_m, observer_xy)

    upd = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((4, 4)),
        dt=1.0,
        n_eff_threshold=0.0,  # disable resampling so weights reflect updates only
        negative_info=negative_info,
    )
    chan = _make_channel(sensor_range_m, observer_xy)

    step_keys = jax.random.split(jax.random.PRNGKey(0), n_steps)
    for k in step_keys:
        belief = upd(
            belief,
            observations=(chan,),
            action=jnp.zeros((1, 2)),
            side=Side.GUARD,
            key=k,
        )
    return initial_inside_mass, _inside_gate_mass(belief, sensor_range_m, observer_xy)


def test_negative_info_modes_monotonically_reduce_inside_gate_mass():
    initial_off, mass_off = _run_with_mode(Off())
    initial_soft, mass_soft = _run_with_mode(Soft(softness_per_channel=(5.0,)))
    initial_hard, mass_hard = _run_with_mode(Hard())

    # Sanity check: priors are identical across runs (same seed).
    assert jnp.isclose(initial_off, initial_soft)
    assert jnp.isclose(initial_off, initial_hard)
    assert initial_off > 0.05, (
        f"Test prior must place real mass inside the gate to be meaningful; "
        f"got initial_inside_mass={initial_off:.3e}"
    )

    # Off is bit-identical to the prior — no negative-info update fires.
    assert jnp.isclose(mass_off, initial_off, atol=1e-6), (
        f"Off mode must not move mass; got initial={initial_off:.3e}, final={mass_off:.3e}"
    )

    # Monotonic ordering: Off >= Soft >= Hard.
    assert mass_off >= mass_soft, (
        f"Off ({mass_off:.3e}) inside-mass should be >= Soft ({mass_soft:.3e})"
    )
    assert mass_soft >= mass_hard, (
        f"Soft ({mass_soft:.3e}) inside-mass should be >= Hard ({mass_hard:.3e})"
    )
    # Hard should drive inside mass to ~0; require at least an order of
    # magnitude below Off after 10 non-detection steps.
    assert mass_hard < 0.1 * mass_off, (
        f"Hard should crush inside mass to <10% of Off; "
        f"got Hard={mass_hard:.3e}, Off={mass_off:.3e}"
    )
