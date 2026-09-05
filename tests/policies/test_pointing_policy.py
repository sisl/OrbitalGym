"""PointingPolicy fills target_dir from the belief mean."""

import flax
import jax
import jax.numpy as jnp

from orbitalgym.policies.pointing import PointingPolicy, PointingTarget


@flax.struct.dataclass
class _Belief:
    mean: jax.Array


@flax.struct.dataclass
class _Cmd:
    dv: jax.Array
    target_dir: jax.Array


def _inner(policy_state, agent_view, key, t):
    n = agent_view.mean.shape[0]
    return _Cmd(dv=jnp.zeros((n, 3)), target_dir=jnp.zeros((n, 3))), policy_state


def _belief():
    # Two guards at (0, 100, 0) and (0, -100, 0); one bandit at (500, 0, 0).
    rows = jnp.array(
        [
            [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -100.0, 0.0, 0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return _Belief(mean=jnp.broadcast_to(rows[None], (2, 3, 6)))


def test_lady_target_points_to_origin():
    cmd, _ = PointingPolicy(inner=_inner, target=PointingTarget.LADY)(None, _belief(), None, 0.0)
    assert jnp.allclose(cmd.target_dir[0], jnp.array([0.0, -1.0, 0.0]), atol=1e-6)


def test_opponent_target_points_to_nearest_bandit():
    cmd, _ = PointingPolicy(inner=_inner, target=PointingTarget.OPPONENT_BELIEF)(
        None, _belief(), None, 0.0
    )
    expected = jnp.array([500.0, -100.0, 0.0])
    assert jnp.allclose(cmd.target_dir[0], expected / jnp.linalg.norm(expected), atol=1e-6)


def test_teammate_target_points_to_nearest_teammate():
    cmd, _ = PointingPolicy(inner=_inner, target=PointingTarget.TEAMMATE)(
        None, _belief(), None, 0.0
    )
    assert jnp.allclose(cmd.target_dir[0], jnp.array([0.0, -1.0, 0.0]), atol=1e-6)
    assert jnp.allclose(cmd.target_dir[1], jnp.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_hold_target_is_zero():
    cmd, _ = PointingPolicy(inner=_inner, target=PointingTarget.HOLD)(None, _belief(), None, 0.0)
    assert jnp.allclose(cmd.target_dir, 0.0)


@flax.struct.dataclass
class _ParticleBelief:
    mean: jax.Array
    particles: jax.Array


def _ring_belief(n_self: int, ring_radius_m: float, n_particles: int = 64):
    """Observers at the origin; the single opponent is a ring cloud of radius R."""
    phases = jnp.linspace(0.0, 2.0 * jnp.pi, n_particles, endpoint=False)
    ring = jnp.stack(
        [
            ring_radius_m * jnp.cos(phases),
            ring_radius_m * jnp.sin(phases),
            jnp.zeros_like(phases),
            jnp.zeros_like(phases),
            jnp.zeros_like(phases),
            jnp.zeros_like(phases),
        ],
        axis=-1,
    )
    n_total = n_self + 1
    particles = jnp.zeros((n_self, n_total, n_particles, 6))
    particles = particles.at[:, n_self, :, :].set(ring)
    mean = jnp.mean(particles, axis=2)
    return _ParticleBelief(mean=mean, particles=particles)


def test_scan_sweeps_in_the_rt_plane_when_the_belief_is_wide():
    rate = 0.0175
    policy = PointingPolicy(inner=_inner, target=PointingTarget.OPPONENT_SCAN, scan_rate_rad_s=rate)
    belief = _ring_belief(n_self=1, ring_radius_m=2000.0)

    cmd0, _ = policy(None, belief, None, 0.0)
    assert jnp.allclose(cmd0.target_dir[0], jnp.array([1.0, 0.0, 0.0]), atol=1e-5)

    cmd1, _ = policy(None, belief, None, jnp.pi / (2.0 * rate))
    assert jnp.allclose(cmd1.target_dir[0], jnp.array([0.0, 1.0, 0.0]), atol=1e-5)


def test_scan_falls_back_to_the_belief_mean_when_the_belief_is_tight():
    belief = _ring_belief(n_self=1, ring_radius_m=10.0)
    # Shift the tight cloud off the origin so the mean direction is well defined.
    particles = belief.particles.at[:, 1, :, 0].add(500.0)
    belief = _ParticleBelief(mean=jnp.mean(particles, axis=2), particles=particles)

    scan, _ = PointingPolicy(inner=_inner, target=PointingTarget.OPPONENT_SCAN)(
        None, belief, None, 137.0
    )
    tracked, _ = PointingPolicy(inner=_inner, target=PointingTarget.OPPONENT_BELIEF)(
        None, belief, None, 137.0
    )
    assert jnp.allclose(scan.target_dir, tracked.target_dir, atol=1e-6)


def test_scan_staggers_two_observers_by_pi():
    policy = PointingPolicy(inner=_inner, target=PointingTarget.OPPONENT_SCAN)
    belief = _ring_belief(n_self=2, ring_radius_m=2000.0)
    cmd, _ = policy(None, belief, None, 61.0)
    assert jnp.allclose(cmd.target_dir[0], -cmd.target_dir[1], atol=1e-5)
