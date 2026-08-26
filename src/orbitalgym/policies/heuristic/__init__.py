"""Scripted-side adversary gallery — heuristic, evasive, and randomized opponents."""

from __future__ import annotations

from orbitalgym.policies.heuristic.evader import OrthogonalEvader
from orbitalgym.policies.heuristic.jittered import JitteredPolicy
from orbitalgym.policies.heuristic.lead_intercept import LeadInterceptPursuer
from orbitalgym.policies.heuristic.sun_tracker import SunTrackerBlocker

__all__ = [
    "JitteredPolicy",
    "LeadInterceptPursuer",
    "OrthogonalEvader",
    "SunTrackerBlocker",
]
