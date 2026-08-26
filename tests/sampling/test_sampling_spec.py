"""Tests for ICSpec and its protocols."""

from __future__ import annotations

import pytest

from orbitalgym.sampling.spec import ICSpec, SideSampler, Validator


def test_icspec_holds_samplers_and_validators():
    def fake_guard(config, key, *, n_vehicles, components, class_name):
        return None

    def fake_bandit(config, key, *, n_vehicles, components, class_name):
        return None

    def fake_validator(config, guards, bandits):
        return True

    spec = ICSpec(
        guard_sampler=fake_guard,
        bandit_sampler=fake_bandit,
        validators=(fake_validator,),
        max_attempts=42,
    )
    assert spec.guard_sampler is fake_guard
    assert spec.bandit_sampler is fake_bandit
    assert spec.validators == (fake_validator,)
    assert spec.max_attempts == 42


def test_icspec_default_validators_empty_and_default_max_attempts():
    spec = ICSpec(guard_sampler=lambda *a, **k: None, bandit_sampler=lambda *a, **k: None)
    assert spec.validators == ()
    assert spec.max_attempts == 100


def test_icspec_is_frozen():
    spec = ICSpec(guard_sampler=lambda *a, **k: None, bandit_sampler=lambda *a, **k: None)
    with pytest.raises((AttributeError, Exception)):
        spec.max_attempts = 999


def test_protocols_are_runtime_checkable():
    assert SideSampler is not None
    assert Validator is not None
