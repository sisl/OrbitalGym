"""Environment package — symmetric core + single-agent projection."""

from orbitalgym.env.core import EnvState, OrbitalGymEnv
from orbitalgym.env.single_agent import SingleAgentView
from orbitalgym.env.types import (
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
    "OrbitalGymEnv",
    "Side",
    "SingleAgentView",
    "SideOutput",
    "SideTrajectory",
    "StepOutput",
    "Trajectory",
]
