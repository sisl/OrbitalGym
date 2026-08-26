"""Tests for observations/negative_info.py — mode types and softness helper."""

from __future__ import annotations

import pytest

from orbitalgym.observations.negative_info import (
    Hard,
    NegativeInfoMode,
    Off,
    Soft,
    softness_from_sigma,
)


def test_off_is_negative_info_mode():
    assert isinstance(Off(), NegativeInfoMode)


def test_hard_is_negative_info_mode():
    assert isinstance(Hard(), NegativeInfoMode)


def test_soft_is_negative_info_mode():
    assert isinstance(Soft(softness_per_channel=(1.0,)), NegativeInfoMode)


def test_soft_rejects_empty_softness():
    with pytest.raises(ValueError, match="non-empty"):
        Soft(softness_per_channel=())


def test_soft_rejects_zero_softness():
    with pytest.raises(ValueError, match="positive"):
        Soft(softness_per_channel=(0.0,))


def test_soft_rejects_negative_softness():
    with pytest.raises(ValueError, match="positive"):
        Soft(softness_per_channel=(1.0, -2.0))


def test_soft_accepts_multi_channel_softness():
    s = Soft(softness_per_channel=(100.0, 0.05))
    assert s.softness_per_channel == (100.0, 0.05)


def test_softness_from_sigma_default_ratio():
    assert softness_from_sigma(50.0) == 50.0


def test_softness_from_sigma_custom_ratio():
    assert softness_from_sigma(50.0, ratio=3.0) == 150.0


def test_off_instances_are_equal():
    """Frozen dataclasses with no fields are equal by value."""
    assert Off() == Off()


def test_hard_instances_are_equal():
    assert Hard() == Hard()


def test_negative_info_types_are_reexported_from_observations_package():
    """Users should be able to `from orbitalgym.observations import Off, Hard, Soft`."""
    from orbitalgym.observations import (
        Hard,
        NegativeInfoMode,
        Off,
        Soft,
        softness_from_sigma,
    )

    assert Off is not None
    assert Hard is not None
    assert Soft is not None
    assert NegativeInfoMode is not None
    assert softness_from_sigma is not None
