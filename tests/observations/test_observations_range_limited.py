"""Tests for RangeLimitedObservation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.env.types import Side
from orbital_game.observations.range_limited import RangeLimitedObservation


class _Layout:
    def __init__(self, n_guards, n_bandits, d):
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


class _Env:
    def __init__(self, guards_rtn, bandits_rtn):
        from types import SimpleNamespace

        self.guards = SimpleNamespace(rtn=guards_rtn)
        self.bandits = SimpleNamespace(rtn=bandits_rtn)


def test_range_limited_visible_only_for_opposing_pairs_in_range():
    """1 guard at origin; 3 bandits at distances 100, 600, 2000 m. Range 1000."""
    n_guards = 1
    n_bandits = 3
    d = 6
    guards_rtn = jnp.zeros((n_guards, d))
    bandits_rtn = jnp.zeros((n_bandits, d))
    bandits_rtn = bandits_rtn.at[0, :3].set(jnp.array([100.0, 0.0, 0.0]))
    bandits_rtn = bandits_rtn.at[1, :3].set(jnp.array([600.0, 0.0, 0.0]))
    bandits_rtn = bandits_rtn.at[2, :3].set(jnp.array([2000.0, 0.0, 0.0]))
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)

    fn = RangeLimitedObservation(layout=layout, sensor_range_m=1000.0, sigma_range=0.0)
    out = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )

    assert len(out) == 1
    ch = out[0]
    assert ch.visible.shape == (n_guards, n_guards + n_bandits)
    assert bool(ch.visible[0, 0]) is False
    assert bool(ch.visible[0, 1]) is True
    assert bool(ch.visible[0, 2]) is True
    assert bool(ch.visible[0, 3]) is False


def test_range_limited_h_matrix_selects_position_rows():
    n_guards = 1
    n_bandits = 1
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    env = _Env(guards_rtn=jnp.zeros((n_guards, d)), bandits_rtn=jnp.zeros((n_bandits, d)))

    fn = RangeLimitedObservation(layout=layout, sensor_range_m=1000.0, sigma_range=1.0)
    out = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )
    ch = out[0]

    expected_h = jnp.eye(3, d)
    assert ch.obs_matrix.shape == (3, d)
    assert jnp.allclose(ch.obs_matrix, expected_h)
    assert jnp.allclose(ch.obs_noise, jnp.eye(3))


def test_range_limited_obs_at_in_range_pair_equals_target_position_when_no_noise():
    n_guards = 1
    n_bandits = 1
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    guards_rtn = jnp.zeros((n_guards, d))
    bandits_rtn = jnp.zeros((n_bandits, d))
    bandits_rtn = bandits_rtn.at[0, :3].set(jnp.array([50.0, 30.0, 10.0]))
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)

    fn = RangeLimitedObservation(layout=layout, sensor_range_m=1000.0, sigma_range=0.0)
    out = fn(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )
    ch = out[0]

    assert jnp.allclose(ch.obs[0, 1], jnp.array([50.0, 30.0, 10.0]))


def test_range_limited_attaches_visibility_score_fn():
    """RangeLimitedObservation must attach a closure that returns
    sensor_range_m - distance for each particle relative to its observer."""
    n_guards = 1
    n_bandits = 1
    d = 6
    layout = _Layout(n_guards=n_guards, n_bandits=n_bandits, d=d)
    # Guard at origin, bandit at (5000, 0, 0) — out of range (irrelevant for
    # the score-fn check below, but matches the spec's geometry).
    guards_rtn = jnp.zeros((n_guards, d))
    bandits_rtn = jnp.zeros((n_bandits, d))
    bandits_rtn = bandits_rtn.at[0, :3].set(jnp.array([5000.0, 0.0, 0.0]))
    env = _Env(guards_rtn=guards_rtn, bandits_rtn=bandits_rtn)

    sensor = RangeLimitedObservation(layout=layout, sensor_range_m=1000.0, sigma_range=1.0)
    (channel,) = sensor(
        env_state=env,
        actions=None,
        side=Side.GUARD,
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
    )

    assert channel.visibility_score_fn is not None

    # Particles tensor shape: (N_obs=1, N_total=2, K=1, d=6).
    # Index 0 in the target axis is the self-pair (guard observing guard).
    # Index 1 is the cross-pair (guard observing bandit).
    particles = jnp.zeros((1, 2, 1, d))
    particles = particles.at[0, 1, 0, :3].set(jnp.array([1500.0, 0.0, 0.0]))
    scores = channel.visibility_score_fn(particles)
    assert scores.shape == (1, 2, 1)
    # Bandit-pair particle at distance 1500 m -> score = 1000 - 1500 = -500.
    assert jnp.allclose(scores[0, 1, 0], -500.0, atol=1e-6)
    # Self-pair particle at origin -> score = 1000 - 0 = +1000.
    assert jnp.allclose(scores[0, 0, 0], 1000.0, atol=1e-6)
