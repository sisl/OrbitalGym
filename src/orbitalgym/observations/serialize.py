"""Versioned observation configuration, with config-derived layouts omitted.

Historical key-only payloads did not retain sensor parameters and cannot be
reconstructed faithfully. ScenarioConfig preserves their old FullObservation
fallback; all new payloads retain constructors and fail on unsupported values.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass

import jax
import jax.numpy as jnp

from orbitalgym.registry import ObservationFnKey, resolve, resolve_class_to_key
from orbitalgym.sampling.serialize import _value_to_primitive, serializable_from_primitive


def observation_to_primitive(sensor):
    """Serialize a registered sensor and recursively nested composites."""
    if not is_dataclass(sensor):
        raise TypeError(f"Observation {type(sensor).__name__} must be a registered dataclass")
    key, enum = resolve_class_to_key(type(sensor))
    if enum is not ObservationFnKey:
        raise TypeError(f"{type(sensor).__name__} is not registered as an observation")
    parameters = {}
    for field in fields(sensor):
        if not field.init or field.name == "layout":
            continue
        value = getattr(sensor, field.name)
        # The unbounded cone is the constructor default. Omitting it preserves
        # the hardware without adding non-standard JSON Infinity tokens.
        if (
            key == ObservationFnKey.CONICAL.value
            and field.name == "max_range_m"
            and value == math.inf
        ):
            continue
        parameters[field.name] = (
            [observation_to_primitive(child) for child in value]
            if key == ObservationFnKey.COMPOSITE.value and field.name == "constituents"
            else (
                {"_array": value.tolist(), "dtype": str(value.dtype)}
                if isinstance(value, jax.Array)
                else _value_to_primitive(value)
            )
        )
    payload = {"_key": key, "_observation_version": 1, "parameters": parameters}
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Observation {type(sensor).__name__} has unsupported constructor values"
        ) from error
    return payload


def observation_from_primitive(payload, layout):
    """Restore a new payload against the newly derived scenario layout."""
    envelope_fields = {"_key", "_observation_version", "parameters"}
    unexpected_fields = payload.keys() - envelope_fields
    if unexpected_fields:
        raise ValueError(f"Unexpected observation envelope fields: {sorted(unexpected_fields)}")
    if payload.get("_observation_version") != 1:
        raise ValueError("Unsupported observation configuration version")
    key = ObservationFnKey(payload["_key"])
    cls = resolve(key)
    if not isinstance(cls, type) or not is_dataclass(cls):
        raise TypeError(f"Observation {key.value} must resolve to a registered dataclass")
    parameters = dict(payload["parameters"])
    allowed = {f.name for f in fields(cls) if f.init and f.name != "layout"}
    unexpected = parameters.keys() - allowed
    if unexpected:
        raise ValueError(f"Unexpected parameters for observation {key.value}: {sorted(unexpected)}")
    for name, value in parameters.items():
        if isinstance(value, dict) and set(value) == {"_array", "dtype"}:
            requested_dtype = jnp.dtype(value["dtype"])
            if jax.dtypes.canonicalize_dtype(requested_dtype) != requested_dtype:
                raise ValueError(
                    f"Observation parameter {name!r} requires {requested_dtype}; "
                    "enable JAX x64 precision before loading this configuration"
                )
            parameters[name] = jnp.asarray(value["_array"], dtype=requested_dtype)
    if key is ObservationFnKey.COMPOSITE:
        parameters["constituents"] = tuple(
            observation_from_primitive(child, layout) for child in parameters["constituents"]
        )
    if any(f.name == "layout" for f in fields(cls)):
        parameters["layout"] = layout
    return serializable_from_primitive({"_key": key.value, **parameters}, ObservationFnKey)
