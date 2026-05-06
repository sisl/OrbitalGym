"""Test helpers for the policies suite.

Builds Command pytree classes for tests that don't have an env handy.
"""

from __future__ import annotations

from orbital_game.actions.assemble import build_command_class
from orbital_game.actions.components import ImpulsiveManeuver
from orbital_game.dynamics.hcw import hcw_rtn_step
from orbital_game.registry import Frame


def make_impulsive_maneuver_command_cls(n_vehicles: int, name: str = "TestCmd"):
    """Return an ImpulsiveManeuver-only Command class for `n_vehicles` agents.

    Defaults to RTN action_frame so the dv field is 3-D (matches the existing
    heuristic-policy tests). Build a separate instance with action_frame=Frame.RT
    when testing 2-D scenarios.
    """
    inst = ImpulsiveManeuver(
        truth_dynamics=hcw_rtn_step,
        action_frame=Frame.RTN,
        truth_frame=Frame.RTN,
        track_mass=False,
    )
    return build_command_class((inst,), n_agents=n_vehicles, class_name=name)
