"""Config JSON must retain observation hardware, including nested composites."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym import make_lady_bandit_guard
from orbitalgym.config import ScenarioConfig
from orbitalgym.env.types import Side
from orbitalgym.observations.composite import CompositeObservation
from orbitalgym.observations.conical import ConicalObservation
from orbitalgym.observations.onboard_gps import OnboardGPSObservation
from orbitalgym.observations.range_limited import RangeLimitedObservation
from orbitalgym.observations.reference import FullObservation
from orbitalgym.observations.teammate_ephemeris import TeammateEphemerisObservation
from tests.observations.test_observations_conical import _state_rtn


def test_default_unbounded_cone_observation_payload_is_strict_json():
    from orbitalgym.observations.serialize import (
        observation_from_primitive,
        observation_to_primitive,
    )

    layout = make_lady_bandit_guard().layout
    sensor = ConicalObservation(layout, jnp.array([[1.0, 0.0, 0.0]]), 0.5)
    payload = observation_to_primitive(sensor)
    encoded = json.dumps(payload, allow_nan=False)
    restored = observation_from_primitive(json.loads(encoded), layout)
    assert restored.max_range_m == float("inf")


def test_nested_composite_config_restores_identical_channels_and_particle_scores():
    cfg = make_lady_bandit_guard(n_guards=2)
    cone = ConicalObservation(
        cfg.layout, jnp.array([[1.0, 0.0, 0.0]]), 0.5, 2.0, 0.04, max_range_m=50.0
    )
    sensor = CompositeObservation(
        (
            cone,
            OnboardGPSObservation(cfg.layout, 0.7),
            CompositeObservation(
                (
                    TeammateEphemerisObservation(cfg.layout, 3.0),
                    RangeLimitedObservation(cfg.layout, 15.0, 0.2),
                    FullObservation(cfg.layout),
                )
            ),
        )
    )
    cfg = dataclasses.replace(cfg, guard_observation_fn=sensor)
    restored = ScenarioConfig.from_json(cfg.to_json())
    assert isinstance(restored.guard_observation_fn, CompositeObservation)
    state = _state_rtn(
        jnp.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]]),
        jnp.array([[10.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
    )

    def call(fn):
        return fn(state, None, Side.GUARD, None, jax.random.PRNGKey(30), 0.0)

    original, recovered = call(sensor), call(restored.guard_observation_fn)
    assert len(original) == len(recovered) == 5
    particles = jnp.ones((2, 4, 4, 6)) * 20.0
    for a, b in zip(original, recovered, strict=True):
        for name in ["obs", "visible", "obs_matrix", "obs_noise"]:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        if a.visibility_score_fn is not None:
            np.testing.assert_array_equal(
                a.visibility_score_fn(particles), b.visibility_score_fn(particles)
            )
    assert restored.guard_observation_fn.constituents[0].layout is restored.layout


def test_historical_key_only_sensor_payload_keeps_legacy_default():
    raw = json.loads(make_lady_bandit_guard().to_json())
    raw["guard_observation_fn"] = {"_key": "conical_observation"}
    assert isinstance(
        ScenarioConfig.from_json(json.dumps(raw)).guard_observation_fn, FullObservation
    )


def test_new_unknown_sensor_payload_fails_clearly():
    raw = json.loads(make_lady_bandit_guard().to_json())
    raw["guard_observation_fn"] = {
        "_key": "unknown_sensor",
        "_observation_version": 1,
        "parameters": {},
    }
    with pytest.raises((KeyError, ValueError), match="unknown_sensor"):
        ScenarioConfig.from_json(json.dumps(raw))


def test_new_payload_revalidates_range_and_rejects_unknown_constructor_fields():
    cfg = make_lady_bandit_guard()
    sensor = ConicalObservation(cfg.layout, jnp.array([[1.0, 0.0, 0.0]]), 0.5)
    raw = json.loads(dataclasses.replace(cfg, guard_observation_fn=sensor).to_json())
    raw["guard_observation_fn"]["parameters"]["max_range_m"] = -1.0
    with pytest.raises(ValueError, match="max_range_m"):
        ScenarioConfig.from_json(json.dumps(raw))
    raw["guard_observation_fn"]["parameters"]["max_range_m"] = 500.0
    raw["guard_observation_fn"]["parameters"]["misspelled_range"] = 500.0
    with pytest.raises(ValueError, match="misspelled_range"):
        ScenarioConfig.from_json(json.dumps(raw))


def test_sensor_array_precision_survives_config_roundtrip():
    cfg = make_lady_bandit_guard()
    sensor = ConicalObservation(
        cfg.layout,
        jnp.array([[1.0, 0.01, 0.0]], dtype=jnp.float32),
        jnp.array([0.123456], dtype=jnp.float32),
        max_range_m=800.0,
    )
    restored = ScenarioConfig.from_json(
        dataclasses.replace(cfg, guard_observation_fn=sensor).to_json()
    )
    assert (
        restored.guard_observation_fn.sensor_boresights_body.dtype
        == sensor.sensor_boresights_body.dtype
    )
    assert restored.guard_observation_fn.half_angle_rad.dtype == sensor.half_angle_rad.dtype


def test_float64_sensor_load_fails_without_mutating_disabled_x64():
    from orbitalgym.observations.serialize import (
        observation_from_primitive,
        observation_to_primitive,
    )

    cfg = make_lady_bandit_guard()
    sensor = ConicalObservation(cfg.layout, jnp.array([[1.0, 0.01, 0.0]], dtype=jnp.float64), 0.5)
    payload = json.loads(json.dumps(observation_to_primitive(sensor)))
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="float64.*x64|x64.*float64"):
            observation_from_primitive(payload, cfg.layout)
        assert not jax.config.x64_enabled
    assert jax.config.x64_enabled


def test_versioned_observation_rejects_unknown_envelope_fields():
    raw = json.loads(make_lady_bandit_guard().to_json())
    raw["guard_observation_fn"]["unexpected_hardware_override"] = 900.0
    with pytest.raises(ValueError, match="unexpected_hardware_override"):
        ScenarioConfig.from_json(json.dumps(raw))
