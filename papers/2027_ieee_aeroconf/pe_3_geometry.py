# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""PE-3: geometry. Initial separation by game horizon at three operating points.

An operating point is a pair (thrust ratio, budget ratio): contested (1, 0.75),
thrust advantage (0.5, 1), and budget advantage (1, 0.5).

Runs:
    pe_3       six separations by six horizons for every pursuer against four
               evaders at the three operating points (3,024 configurations)
    pe_3_fine  both policy groups on 26 separations by 31 horizons against a
               coasting evader at (1, 0.75) and a fleeing evader at (1, 0.5),
               for the contour maps of Figure 5

Run on a GPU from the repository root (about 30 min on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/pe_3_geometry.py

Name one or more runs to execute only those, for example ``pe_3_geometry.py pe_3``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools

import numpy as np
from baseline import FOCUS_EVADERS, GROUP_PURSUERS, OPERATING_POINTS, PURSUERS, pe_game
from runner import Run, main

SEPARATIONS = (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
HORIZONS = (300.0, 600.0, 900.0, 1800.0, 3600.0, 5400.0)
FINE_SEPARATIONS = tuple(float(v) for v in np.round(100.0 * 2.0 ** np.linspace(0, 5, 26), 3))
FINE_HORIZONS = tuple(float(v) for v in np.linspace(300.0, 1800.0, 31))
FINE_ROWS = (("coast", 1.0, 0.75), ("flee", 1.0, 0.5))


def grid():
    """Separation by horizon for every pursuer, four evaders, and three operating points."""
    return [
        pe_game(thrust, fuel, pursuer=p, evader=e, separation_m=d, horizon_s=t)
        for (thrust, fuel), p, e, d, t in itertools.product(
            OPERATING_POINTS, PURSUERS, FOCUS_EVADERS, SEPARATIONS, HORIZONS
        )
    ]


def fine_grid():
    """Both policy groups on the fine separation by horizon grid at two operating points."""
    return [
        pe_game(thrust, fuel, pursuer=p, evader=e, separation_m=d, horizon_s=t)
        for e, thrust, fuel in FINE_ROWS
        for p, d, t in itertools.product(GROUP_PURSUERS, FINE_SEPARATIONS, FINE_HORIZONS)
    ]


RUNS = (Run("pe_3", grid), Run("pe_3_fine", fine_grid))


if __name__ == "__main__":
    main(RUNS, __doc__)
