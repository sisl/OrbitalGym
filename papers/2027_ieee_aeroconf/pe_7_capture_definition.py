# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""PE-7: capture definition. Capture radius, velocity-matching condition, and capture test.

Runs:
    pe_7           four capture radii by three velocity-matching conditions
                   (none, 2 m/s, 0.5 m/s) for six pursuers against four evaders
                   at three operating points (864 configurations)
    pe_7_endpoint  the capability sweep of PE-1 with capture tested only at the
                   end of each step; planner rollouts keep the ten-point test
                   (3,780 configurations)

Run on a GPU from the repository root (about 20 min on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/pe_7_capture_definition.py

Name one or more runs to execute only those, for example ``pe_7_capture_definition.py pe_7``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools
from dataclasses import replace

import numpy as np
from baseline import FOCUS_EVADERS, GROUP_PURSUERS, OPERATING_POINTS, capability_grid, pe_game
from pe_game import Planner
from runner import Run, main


def grid():
    """Capture radius by velocity-matching condition at the three operating points."""
    return [
        pe_game(thrust, fuel, pursuer=p, evader=e, capture_m=radius, capture_speed_mps=speed)
        for (thrust, fuel), p, e, radius, speed in itertools.product(
            OPERATING_POINTS,
            GROUP_PURSUERS,
            FOCUS_EVADERS,
            (5.0, 10.0, 20.0, 50.0),
            (np.inf, 2.0, 0.5),
        )
    ]


def endpoint_grid():
    """PE-1 with the game testing capture once per step instead of at ten points."""
    return [(replace(g, substeps=1), Planner(rollout_substeps=10)) for g in capability_grid()]


RUNS = (Run("pe_7", grid), Run("pe_7_endpoint", endpoint_grid))


if __name__ == "__main__":
    main(RUNS, __doc__)
