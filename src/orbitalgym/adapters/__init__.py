"""Framework adapters — thin wrappers at the host-array boundary.

`GymnasiumAdapter` wraps `OrbitalGymEnv` as a `gymnasium.Env` for
single-agent RL. `PettingZooAdapter` wraps it as a `pettingzoo.ParallelEnv`
for multi-agent RL. `POMDPAdapter` is a duck-typed POMDPPlanners-shape
projection (no external import).
"""

from __future__ import annotations

from orbitalgym.adapters.gymnasium import GymnasiumAdapter
from orbitalgym.adapters.pettingzoo import PettingZooAdapter
from orbitalgym.adapters.pomdp import POMDPAdapter

__all__ = ["GymnasiumAdapter", "PettingZooAdapter", "POMDPAdapter"]
