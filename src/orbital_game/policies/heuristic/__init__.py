"""Scripted-side adversary gallery — heuristic, evasive, and randomized opponents."""

from __future__ import annotations

from orbital_game.policies.heuristic.evader import OrthogonalEvader
from orbital_game.policies.heuristic.jittered import JitteredPolicy
from orbital_game.policies.heuristic.lead_intercept import LeadInterceptPursuer
from orbital_game.policies.heuristic.sun_tracker import SunTrackerBlocker

__all__ = [
    "JitteredPolicy",
    "LeadInterceptPursuer",
    "OrthogonalEvader",
    "SunTrackerBlocker",
]
