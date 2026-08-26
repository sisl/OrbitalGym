"""Tests for Observation dataclass and flatten_observations helper."""

from __future__ import annotations

import jax.numpy as jnp

from orbitalgym.observations.types import Observation, flatten_observations


def test_observation_construction_with_default_obs_fn_none():
    obs = Observation(
        obs=jnp.zeros((2, 3, 6)),
        visible=jnp.ones((2, 3), dtype=bool),
        obs_matrix=jnp.eye(6),
        obs_noise=jnp.eye(6) * 0.01,
    )
    assert obs.obs.shape == (2, 3, 6)
    assert obs.visible.shape == (2, 3)
    assert obs.obs_matrix.shape == (6, 6)
    assert obs.obs_noise.shape == (6, 6)
    assert obs.obs_fn is None


def test_observation_with_nonlinear_obs_fn():
    def h(x):
        return jnp.array([jnp.linalg.norm(x[:3])])

    obs = Observation(
        obs=jnp.zeros((1, 1, 1)),
        visible=jnp.ones((1, 1), dtype=bool),
        obs_matrix=jnp.zeros((1, 6)),
        obs_noise=jnp.eye(1) * 0.01,
        obs_fn=h,
    )
    assert obs.obs_fn is h


def test_flatten_observations_concatenates_channel_obs():
    a = Observation(
        obs=jnp.array([[[1.0, 2.0], [3.0, 4.0]]]),  # (1, 2, 2)
        visible=jnp.ones((1, 2), dtype=bool),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
    )
    b = Observation(
        obs=jnp.array([[[5.0], [6.0]]]),  # (1, 2, 1)
        visible=jnp.ones((1, 2), dtype=bool),
        obs_matrix=jnp.eye(1, 2),
        obs_noise=jnp.eye(1),
    )
    flat = flatten_observations((a, b))
    assert flat.shape == (6,)
    assert jnp.allclose(flat, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_flatten_observations_empty_tuple_raises():
    import pytest

    with pytest.raises(ValueError, match="at least one channel"):
        flatten_observations(())


def test_observation_has_visibility_score_fn_field_defaulting_to_none():
    """Observation gains an optional visibility_score_fn field for PF
    negative-information updates. Default None preserves backward compatibility."""
    o = Observation(
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[True]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
    )
    assert o.visibility_score_fn is None


def test_observation_accepts_visibility_score_fn():
    def score(particles):
        return jnp.zeros(particles.shape[:-1])

    o = Observation(
        obs=jnp.zeros((1, 1, 2)),
        visible=jnp.array([[True]]),
        obs_matrix=jnp.eye(2),
        obs_noise=jnp.eye(2),
        visibility_score_fn=score,
    )
    assert o.visibility_score_fn is score
