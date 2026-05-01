"""Tests for ICSpec and its protocols."""

from __future__ import annotations

import pytest

from orbital_game.sampling.spec import ICSpec, SideSampler, Validator


def test_icspec_holds_samplers_and_validators():
    def fake_def(config, key, *, n_vehicles, components, class_name):
        return None

    def fake_int(config, key, *, n_vehicles, components, class_name):
        return None

    def fake_validator(config, defs, ints):
        return True

    spec = ICSpec(
        defender_sampler=fake_def,
        intruder_sampler=fake_int,
        validators=(fake_validator,),
        max_attempts=42,
    )
    assert spec.defender_sampler is fake_def
    assert spec.intruder_sampler is fake_int
    assert spec.validators == (fake_validator,)
    assert spec.max_attempts == 42


def test_icspec_default_validators_empty_and_default_max_attempts():
    spec = ICSpec(defender_sampler=lambda *a, **k: None, intruder_sampler=lambda *a, **k: None)
    assert spec.validators == ()
    assert spec.max_attempts == 100


def test_icspec_is_frozen():
    spec = ICSpec(defender_sampler=lambda *a, **k: None, intruder_sampler=lambda *a, **k: None)
    with pytest.raises((AttributeError, Exception)):
        spec.max_attempts = 999


def test_protocols_are_runtime_checkable():
    assert SideSampler is not None
    assert Validator is not None
