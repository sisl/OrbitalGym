# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""PE-5: lookahead. The MPPI-Flee pursuer's planning lookahead by the quality of its information.

At the contested operating point (equal thrust limits, budget ratio 0.75), the
pursuer plans 90 to 900 s ahead with full information or with a 5 km radial
sensor at 0.1% or 1% noise. Only the pursuer senses, and an MPPI evader keeps
the default 600 s lookahead.

Runs:
    pe_5  five lookaheads by three information levels against three evaders
          (45 configurations), for Figure 7

Run on a GPU from the repository root (about 3 min on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/pe_5_lookahead.py

Name one or more runs to execute only those, for example ``pe_5_lookahead.py pe_5``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools

from baseline import pe_game
from pe_game import Planner
from runner import Run, main

SEGMENTS = (3, 7, 12, 20, 30)
INFORMATION = (
    {},
    dict(sensor="radial", sensor_range_m=5000.0),
    dict(sensor="radial", sensor_range_m=5000.0, noise_fraction=1e-3),
)


def grid():
    """A lookahead is ``segments`` impulses, each held for three 10 s steps."""
    entries = []
    for segments, sensing, e in itertools.product(
        SEGMENTS, INFORMATION, ("flee", "random", "mppi_lqr")
    ):
        if sensing:
            sensing = dict(sensing, sensed="pursuer", prior_sigma_m=30.0, prior_sigma_mps=0.01)
        game = pe_game(1.0, 0.75, pursuer="mppi_flee", evader=e, **sensing)
        entries.append((game, Planner(segments=segments, evader_segments=Planner().segments)))
    return entries


RUNS = (Run("pe_5", grid),)


if __name__ == "__main__":
    main(RUNS, __doc__)
