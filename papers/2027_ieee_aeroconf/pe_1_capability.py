# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""PE-1: capability. Every pursuer against every evader over the thrust ratio and the budget ratio.

The pursuer has a thrust limit of 0.1 m/s per step and a 5 m/s delta-v budget;
the encounter starts 300 m apart and lasts 1,800 s. The ratios divide the
evader's thrust limit and budget by the pursuer's.

Runs:
    pe_1        the 9 by 10 grid of ratios of Table 4 (3,780 configurations)
    pe_1_fine   both policy groups against the fleeing evader on a 31 by 31
                grid of ratios, for the contour maps of Figure 3(d-f)
    pe_1_seed2  pe_1 on a second, independent set of 128 initial conditions
    pe_1_n512   pe_1 on 512 new encounters per configuration

Run on a GPU from the repository root (about 2 h for all runs, 13 min for pe_1,
on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/pe_1_capability.py

Name one or more runs to execute only those, for example ``pe_1_capability.py pe_1``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

from baseline import capability_grid, fine_capability_grid
from runner import Run, main


def fine_grid():
    """Both policy groups against the fleeing evader on the fine grid, 5 m/s pursuer budget."""
    return fine_capability_grid((5.0,))


RUNS = (
    Run("pe_1", capability_grid),
    Run("pe_1_fine", fine_grid),
    Run("pe_1_seed2", capability_grid, seed=777),
    Run("pe_1_n512", capability_grid, episodes=512, seed=20261001, batch_cells=45),
)


if __name__ == "__main__":
    main(RUNS, __doc__)
