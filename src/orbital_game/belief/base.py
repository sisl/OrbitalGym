"""Belief protocols."""

from __future__ import annotations

from typing import Any, Protocol


class Belief(Protocol):
    """Marker — concrete beliefs are typed pytrees."""


class BeliefInitializer(Protocol):
    def __call__(self, config, env_state, key) -> Any: ...


class BeliefUpdater(Protocol):
    def __call__(self, belief, action, observation, params, dt, key) -> Any: ...
