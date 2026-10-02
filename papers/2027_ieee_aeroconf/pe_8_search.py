# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""PE-8: search. Monte Carlo tree search on both sides, and longer HCW intercept lookaheads.

Runs:
    pe_8                MCTS as pursuer against five evaders and as evader
                        against five pursuers on a 5 by 5 subset of the ratios
                        of PE-1 (250 configurations), for Table 6
    pe_8_hcw_lookahead  HCW intercept with 300 and 600 s lookaheads against
                        the six evaders over the grid of PE-1 (1,080
                        configurations)

Run on a GPU from the repository root (about 40 min on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/pe_8_search.py

Name one or more runs to execute only those, for example ``pe_8_search.py pe_8``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools

import numpy as np
from baseline import BUDGET_RATIOS, EVADERS, RATIOS, pe_game
from runner import Run, main

THRUST_SUBSET = (0.5, 0.75, 1.0, 1.25, 2.0)
BUDGET_SUBSET = (0.5, 0.75, 0.9, 1.0, np.inf)


def grid():
    """MCTS as the pursuer, then as the evader, on the subset of ratios."""
    games = [
        pe_game(thrust, fuel, pursuer="mcts", evader=e)
        for e, thrust, fuel in itertools.product(
            ("coast", "random", "flee", "mppi_direct", "mppi_lqr"), THRUST_SUBSET, BUDGET_SUBSET
        )
    ]
    games += [
        pe_game(thrust, fuel, pursuer=p, evader="mcts")
        for p, thrust, fuel in itertools.product(
            ("lqr", "hcw", "mppi_coast", "mppi_flee", "mcts"), THRUST_SUBSET, BUDGET_SUBSET
        )
    ]
    return games


def hcw_lookahead_grid():
    """HCW intercept with lookaheads of 300 and 600 s over the grid of PE-1."""
    return [
        pe_game(thrust, fuel, pursuer="hcw", evader=e, lookahead_s=lookahead)
        for lookahead in (300.0, 600.0)
        for e, thrust, fuel in itertools.product(EVADERS, RATIOS, BUDGET_RATIOS)
    ]


RUNS = (Run("pe_8", grid), Run("pe_8_hcw_lookahead", hcw_lookahead_grid))


if __name__ == "__main__":
    main(RUNS, __doc__)
