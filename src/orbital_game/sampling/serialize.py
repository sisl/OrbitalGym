"""Generic frozen-dataclass-with-enum-key serializer.

Any frozen dataclass registered via @register(SomeKey.X) round-trips through
serializable_to_primitive / serializable_from_primitive without per-class JSON
branches in config.py.

Supported field types: jax.Array (-> nested list), Enum (-> .value), bool / int
/ float / str / None / tuple / list. Nested registered dataclasses serialize
recursively; the caller passes the enum class to use for each top-level call.
"""

from __future__ import annotations

import dataclasses as _dc
import inspect
import typing
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, cast

import jax
import jax.numpy as jnp

from orbital_game.registry import resolve, resolve_class_to_key


def serializable_to_primitive(obj: Any) -> dict:
    """Serialize a registered frozen dataclass to a JSON-friendly dict.

    The output dict contains '_key' (the registered enum value) plus one entry
    per dataclass field. jax.Array fields become lists; Enum fields become
    .value; nested registered dataclasses serialize recursively.
    """
    if not is_dataclass(obj):
        raise TypeError(f"Expected a dataclass instance, got {type(obj).__name__}")
    key_value, _enum_class = resolve_class_to_key(type(obj))
    out: dict = {"_key": key_value}
    for f in fields(obj):
        out[f.name] = _value_to_primitive(getattr(obj, f.name))
    return out


def serializable_from_primitive(d: dict, key_enum_cls: type[Enum]) -> Any:
    """Reconstruct a registered frozen dataclass from its serialized form.

    The '_key' entry is looked up in `key_enum_cls`'s registered classes;
    remaining entries are passed by name to the class constructor. Per-field
    type rehydration (list -> jax.Array, scalar -> Enum) is done by inspecting
    each field's resolved annotation on the resolved class.
    """
    payload = dict(d)
    key_value = payload.pop("_key")
    try:
        key = key_enum_cls(key_value)
    except ValueError as e:
        raise KeyError(
            f"No callable registered for {key_value!r} (not a valid {key_enum_cls.__name__})"
        ) from e
    resolved = resolve(key)
    if not (is_dataclass(resolved) and isinstance(resolved, type)):
        raise TypeError(f"Resolved class for {key_value!r} is not a dataclass class")
    cls = cast(type, resolved)
    hints = typing.get_type_hints(cls)
    kwargs: dict = {}
    for f in fields(cls):
        if f.name not in payload:
            if f.default is _dc.MISSING and f.default_factory is _dc.MISSING:
                raise KeyError(f"Missing required field {f.name!r} for {cls.__qualname__}")
            continue
        hint = hints.get(f.name, f.type)
        kwargs[f.name] = _value_from_primitive(payload[f.name], hint)
    return cls(**kwargs)


def _value_to_primitive(v: Any) -> Any:
    if isinstance(v, jax.Array):
        return jnp.asarray(v).tolist()
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (tuple, list)):
        return [_value_to_primitive(x) for x in v]
    if is_dataclass(v):
        return serializable_to_primitive(v)
    return v


def _value_from_primitive(v: Any, annotation: Any) -> Any:
    """Rehydrate from primitive form, dispatching on the resolved annotation.

    Handles the cases relevant to sampler/validator dataclasses:
      Enum subclasses              -> annotation(value)
      jax.Array / ndarray          -> jnp.asarray on lists
      None                         -> preserved
    Falls back to a string-substring heuristic for annotations that
    `typing.get_type_hints` could not resolve to a concrete class.
    """
    if v is None:
        return None
    # Concrete type from get_type_hints when possible.
    if inspect.isclass(annotation):
        if issubclass(annotation, Enum):
            return annotation(v)
        is_array_type = "Array" in annotation.__name__ or "ndarray" in annotation.__name__
        if is_array_type and isinstance(v, list):
            return jnp.asarray(v)
    # Fall back to string-annotation heuristic for cases where get_type_hints
    # can't resolve (e.g. Optional[T], unions, forward refs).
    ann_str = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", str(annotation))
    )
    is_array_ann = "jax.Array" in ann_str or "Array" in ann_str or "ndarray" in ann_str
    if is_array_ann and isinstance(v, list):
        return jnp.asarray(v)
    return v
