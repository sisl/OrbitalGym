"""Controlled-side policy gallery — patterns that wrap or compose other policies."""

from __future__ import annotations

from orbitalgym.policies.controlled.composite_action import CompositeActionPolicy
from orbitalgym.policies.controlled.heuristic_with_fallback import HeuristicWithFallbackPolicy

__all__ = [
    "CompositeActionPolicy",
    "HeuristicWithFallbackPolicy",
]
