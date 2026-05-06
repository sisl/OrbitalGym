"""Role-agnostic policy library — shared between guard and bandit.

Public re-exports:

    ZeroControl     — default heuristic policy; emits zero command.
    Policy          — structural protocol.

Submodules:

    controlled      — gallery of controlled-side patterns
                      (scripted_with_fallback, belief_conditioned, composite_action).
    heuristic       — gallery of heuristic policies
                      (lead_intercept, evader, sun_tracker, jittered).
"""

from __future__ import annotations

from orbital_game.policies.base import Policy
from orbital_game.policies.uniform_random import UniformRandomDiscretePolicy
from orbital_game.policies.zero import ZeroControl

__all__ = ["Policy", "UniformRandomDiscretePolicy", "ZeroControl"]
