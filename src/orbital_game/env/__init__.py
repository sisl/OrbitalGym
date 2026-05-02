"""Environment package — symmetric core + single-agent projection."""

from orbital_game.env.core import EnvState, OrbitalGameEnv
from orbital_game.env.single_agent import SingleAgentView
from orbital_game.env.types import (
    Actions,
    BySide,
    Side,
    SideOutput,
    SideTrajectory,
    StepOutput,
    Trajectory,
)

__all__ = [
    "Actions",
    "BySide",
    "EnvState",
    "OrbitalGameEnv",
    "Side",
    "SingleAgentView",
    "SideOutput",
    "SideTrajectory",
    "StepOutput",
    "Trajectory",
]
