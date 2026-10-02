# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""LBG-2: geometry. The bandit's starting distance from the lady by the game horizon.

The guard has a thrust limit of 0.1 m/s per step and a 5 m/s delta-v budget. By
default the bandit starts 1,000 m and the guard 200 m from the lady, the game
lasts 1,800 s, the capture radius is 10 m, and the breach radius is 50 m.

Runs:
    lbg_2  four starting distances by four horizons for every guard against
           every bandit at equal thrust limits and budgets (784 configurations)

Run on a GPU from the repository root (about 5 min on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/lbg_2_geometry.py

Name one or more runs to execute only those, for example ``lbg_2_geometry.py lbg_2``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools

from baseline import lbg_game
from lbg_game import BANDITS, GUARDS
from runner import Run, main


def grid():
    """Bandit starting distance by horizon at equal capability."""
    return [
        lbg_game(1.0, 1.0, guard=g, bandit=b, bandit_separation_m=d, horizon_s=t)
        for g, b, d, t in itertools.product(
            GUARDS, BANDITS, (500.0, 1000.0, 2000.0, 4000.0), (600.0, 1200.0, 1800.0, 3600.0)
        )
    ]


RUNS = (Run("lbg_2", grid),)


if __name__ == "__main__":
    main(RUNS, __doc__)
