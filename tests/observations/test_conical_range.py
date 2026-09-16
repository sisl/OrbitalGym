"""A finite cone is one intersected gate, including its non-detection model."""

import dataclasses
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym import make_lady_bandit_guard
from orbitalgym.belief.pf import ParticleFilterBeliefUpdater
from orbitalgym.config import ScenarioConfig
from orbitalgym.env.types import Side
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.observations.negative_info import Hard
from orbitalgym.observations.types import OUT_OF_SCOPE_SCORE
from tests.belief.test_belief_pf_negative_info import _identity_dynamics, _make_belief
from tests.observations.test_observations_conical import (
    _layout_rt,
    _layout_rtn,
    _state_rt,
    _state_rtn,
)


def _sensor(d=6, **kwargs):
    return ConicalObservation(
        layout=_layout_rt() if d == 4 else _layout_rtn(),
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0]]),
        half_angle_rad=math.pi / 6,
        sigma_floor=0.0,
        **kwargs,
    )


def _observe(sensor, targets):
    d = sensor.layout.dynamics_state_dim
    positions = jnp.asarray(targets, dtype=jnp.float32)
    if d == 4:
        state = _state_rt(
            jnp.zeros((1, 4)), jnp.concatenate([positions, jnp.zeros_like(positions)], axis=1)
        )
    else:
        state = _state_rtn(jnp.zeros((1, 3)), positions)
    return sensor(state, None, Side.GUARD, None, jax.random.PRNGKey(0), 0.0)[0]


@pytest.mark.parametrize("d", [4, 6])
def test_range_and_cone_intersection_jit_and_padded_2d(d):
    positions = np.array([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.01, 0.0, 0.0], [0.0, 5.0, 0.0]])[
        :, : d // 2
    ]
    sensor = _sensor(d, max_range_m=10.0)
    channel = _observe(sensor, positions)
    np.testing.assert_array_equal(channel.visible, [[False, True, True, False, False]])
    particles = jnp.zeros((1, 5, 1, d)).at[0, 1:, 0, : d // 2].set(positions)
    score = jax.jit(channel.visibility_score_fn)(particles)
    assert score[0, 0, 0] == OUT_OF_SCOPE_SCORE
    assert score[0, 1, 0] > 0
    assert score[0, 2, 0] == 0
    assert score[0, 3, 0] < 0 and score[0, 4, 0] < 0
    compiled = jax.jit(lambda: _observe(sensor, positions).visible)()
    np.testing.assert_array_equal(compiled, channel.visible)


def test_unbounded_default_preserves_exact_legacy_scores_and_noise():
    legacy = _observe(_sensor(), [[1.0e8, 0.0, 0.0]])
    explicit = _observe(_sensor(max_range_m=math.inf), [[1.0e8, 0.0, 0.0]])
    assert legacy.visible[0, 1]
    particles = jnp.array([[[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0]]]])
    np.testing.assert_array_equal(legacy.obs, explicit.obs)
    np.testing.assert_array_equal(
        legacy.visibility_score_fn(particles), explicit.visibility_score_fn(particles)
    )
    np.testing.assert_allclose(legacy.visibility_score_fn(particles)[0, 1], math.pi / 6)


@pytest.mark.parametrize("radius", [0.0, -1.0, -math.inf, math.nan])
def test_invalid_range_rejected_at_construction(radius):
    with pytest.raises(ValueError, match="max_range_m"):
        _sensor(max_range_m=radius)


def test_zero_separation_retains_existing_cone_behavior_without_nan():
    channel = _observe(_sensor(max_range_m=10.0), [[0.0, 0.0, 0.0]])
    assert not channel.visible[0, 1]
    scores = channel.visibility_score_fn(jnp.zeros((1, 2, 3, 6)))
    assert np.isfinite(scores).all()
    assert (scores[0, 1] < 0).all()


def test_non_detection_removes_only_particles_inside_both_gates():
    channel = _observe(_sensor(max_range_m=10.0), [[15.0, 0.0, 0.0]])
    # In cone/in range; in cone/out of range; out of cone/in range.
    particles = (
        jnp.zeros((1, 2, 3, 6))
        .at[0, 1, :, :3]
        .set(jnp.array([[5.0, 0.0, 0.0], [15.0, 0.0, 0.0], [0.0, 5.0, 0.0]]))
    )
    belief = _make_belief(particles)
    updater = ParticleFilterBeliefUpdater(
        dynamics_fn=_identity_dynamics,
        process_noise=jnp.zeros((6, 6)),
        dt=1.0,
        n_eff_threshold=0.0,
        negative_info=Hard(),
    )
    updated = jax.jit(
        lambda b: updater(b, (channel,), jnp.zeros((1, 3)), Side.GUARD, jax.random.PRNGKey(1))
    )(belief)
    weights = jax.nn.softmax(updated.log_weights, axis=-1)
    assert weights[0, 1, 0] < 1.0e-6
    np.testing.assert_allclose(weights[0, 1, 1:], [0.5, 0.5], atol=1.0e-6)
    np.testing.assert_allclose(weights[0, 0], [1 / 3] * 3, atol=1.0e-6)


@pytest.mark.parametrize("radius", [math.inf, 800.0])
def test_config_roundtrip_preserves_cone_noise_and_range(radius):
    cfg = make_lady_bandit_guard()
    sensor = dataclasses.replace(
        _sensor(max_range_m=radius),
        layout=cfg.layout,
        sigma_floor=2.0,
        sigma_range_frac=0.03,
        half_angle_rad=(0.4, 0.6),
        sensor_boresights_body=jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )
    cfg = dataclasses.replace(cfg, guard_observation_fn=sensor, bandit_observation_fn=sensor)
    restored = ScenarioConfig.from_json(cfg.to_json())
    for loaded in [restored.guard_observation_fn, restored.bandit_observation_fn]:
        assert isinstance(loaded, ConicalObservation)
        assert loaded.max_range_m == radius
        assert loaded.sigma_floor == 2.0 and loaded.sigma_range_frac == 0.03
        np.testing.assert_allclose(loaded.half_angle_rad, [0.4, 0.6])
        np.testing.assert_array_equal(loaded.sensor_boresights_body, sensor.sensor_boresights_body)
        assert loaded.layout is restored.layout
    raw = json.loads(cfg.to_json())
    raw["guard_observation_fn"]["parameters"].pop("max_range_m", None)
    assert math.isinf(ScenarioConfig.from_json(json.dumps(raw)).guard_observation_fn.max_range_m)


def test_identical_sensor_hardware_yields_geometry_dependent_visibility():
    sensor = _sensor(max_range_m=800.0)
    state = _state_rtn(jnp.array([[-600.0, 0.0, 0.0], [-2200.0, 0.0, 0.0]]), jnp.zeros((1, 3)))
    channel = sensor(state, None, Side.GUARD, None, jax.random.PRNGKey(0), 0.0)[0]
    np.testing.assert_array_equal(channel.visible, [[False, False, True], [False, False, False]])
    particles = jnp.zeros((2, 3, 1, 6))
    scores = channel.visibility_score_fn(particles)
    assert scores[0, 2, 0] > 0 and scores[1, 2, 0] < 0
    assert (scores[:, :2] == OUT_OF_SCOPE_SCORE).all()
