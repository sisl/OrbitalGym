"""Test helpers for the policies suite.

Builds Command pytree classes for tests that don't have an env handy.
"""

from __future__ import annotations

from orbital_game.actions.assemble import build_command_class
from orbital_game.actions.components import ImpulsiveManeuver


def make_impulsive_maneuver_command_cls(n_vehicles: int, name: str = "TestCmd"):
    """Return an ImpulsiveManeuver-only Command class for `n_vehicles` agents.

    Uses ImpulsiveManeuver as the sole action component (the simple `dv`-shaped
    action), which matches every heuristic policy currently in the gallery.
    """
    return build_command_class((ImpulsiveManeuver,), n_agents=n_vehicles, class_name=name)
