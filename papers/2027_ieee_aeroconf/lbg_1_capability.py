# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""LBG-1: capability. Every guard against every bandit over the thrust and budget ratios.

The guard has a thrust limit of 0.1 m/s per step and a 5 m/s delta-v budget. By
default the bandit starts 1,000 m and the guard 200 m from the lady, the game
lasts 1,800 s, the capture radius is 10 m, and the breach radius is 50 m.
The grids are built from the bandit's capability over the guard's, the
reciprocal of the guard-over-bandit ratios that the paper reports.

Runs:
    lbg_1       the 9 by 10 grid of ratios of Table 5 (4,410 configurations)
    lbg_1_fine  every guard against the LQR and MPPI-LQR bandits on a 31 by 31
                grid of ratios, for the contour maps of Figure 4
    lbg_1_n512  lbg_1 on 512 new encounters per configuration

Run on a GPU from the repository root (about 2.5 h for all runs, 17 min for lbg_1,
on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/lbg_1_capability.py

Name one or more runs to execute only those, for example ``lbg_1_capability.py lbg_1``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools

from baseline import BUDGET_RATIOS, FINE_RATIOS, RATIOS, lbg_game
from lbg_game import BANDITS, GUARDS
from runner import Run, main


def grid():
    """Guard by bandit over the bandit-over-guard thrust and budget ratios."""
    return [
        lbg_game(thrust, fuel, guard=g, bandit=b)
        for g, b, thrust, fuel in itertools.product(GUARDS, BANDITS, RATIOS, BUDGET_RATIOS)
    ]


def fine_grid():
    """Every guard against the LQR and MPPI-LQR bandits on the fine grid of ratios."""
    return [
        lbg_game(thrust, fuel, guard=g, bandit=b)
        for b, g, thrust, fuel in itertools.product(
            ("lqr", "mppi_lqr"), GUARDS, FINE_RATIOS, FINE_RATIOS
        )
    ]


RUNS = (
    Run("lbg_1", grid),
    Run("lbg_1_fine", fine_grid),
    Run("lbg_1_n512", grid, episodes=512, seed=20261001, batch_cells=45),
)


if __name__ == "__main__":
    main(RUNS, __doc__)
