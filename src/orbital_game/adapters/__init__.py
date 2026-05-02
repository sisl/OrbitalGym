"""Framework adapters — thin wrappers at the host-array boundary.

Adapters are opt-in: gymnasium and pettingzoo are peer dependencies that
must be installed via the matching pip extra. POMDPAdapter has no external
dependency (POMDPPlanners-shape protocol, not import).

Adapters that fail to import (extra not installed) are silently omitted
from `__all__` rather than raising at package import. To check whether
an adapter is available::

    from orbital_game.adapters import GymnasiumAdapter
    if GymnasiumAdapter is None:
        ...  # pip install orbital-game[gymnasium]

Or import the submodule directly to get a clear ImportError::

    from orbital_game.adapters.gymnasium import GymnasiumAdapter

Available pip extras:
    pip install orbital-game[gymnasium]
    pip install orbital-game[pettingzoo]
"""

from __future__ import annotations

# POMDPAdapter has no external dependency — always available.
from orbital_game.adapters.pomdp import POMDPAdapter

__all__: list[str] = ["POMDPAdapter"]

# Optional adapters: silently None if the extra isn't installed.
try:
    from orbital_game.adapters.gymnasium import GymnasiumAdapter

    __all__ += ["GymnasiumAdapter"]
except ImportError:  # pragma: no cover — gymnasium extra not installed
    GymnasiumAdapter = None  # type: ignore[assignment,misc]

try:
    from orbital_game.adapters.pettingzoo import PettingZooAdapter

    __all__ += ["PettingZooAdapter"]
except ImportError:  # pragma: no cover — pettingzoo extra not installed
    PettingZooAdapter = None  # type: ignore[assignment,misc]
