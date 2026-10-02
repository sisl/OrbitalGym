# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""PE-2: budget size. The capability sweep of PE-1 at a small and a large pursuer delta-v budget.

Runs:
    pe_2       pursuer budgets of 1 and 20 m/s over the grid of ratios of PE-1,
               and unlimited budgets for both sides over the thrust ratio
               (7,938 configurations)
    pe_2_fine  both policy groups against the fleeing evader on a 31 by 31
               grid of ratios at 1 and 20 m/s, for the contour maps of
               Figure 3(a-c, g-i)

Run on a GPU from the repository root (about 1 h on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/pe_2_budget_size.py

Name one or more runs to execute only those, for example ``pe_2_budget_size.py pe_2``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools

import numpy as np
from baseline import EVADERS, PURSUERS, RATIOS, capability_grid, fine_capability_grid, pe_game
from runner import Run, main


def grid():
    """PE-1 at pursuer budgets of 1 and 20 m/s, then unlimited budgets for both sides."""
    games = [game for budget in (1.0, 20.0) for game in capability_grid(budget)]
    games += [
        pe_game(thrust, 1.0, np.inf, pursuer=p, evader=e)
        for p, e, thrust in itertools.product(PURSUERS, EVADERS, RATIOS)
    ]
    return games


def fine_grid():
    """Both policy groups against the fleeing evader on the fine grid at 1 and 20 m/s."""
    return fine_capability_grid((1.0, 20.0))


RUNS = (Run("pe_2", grid), Run("pe_2_fine", fine_grid))


if __name__ == "__main__":
    main(RUNS, __doc__)
