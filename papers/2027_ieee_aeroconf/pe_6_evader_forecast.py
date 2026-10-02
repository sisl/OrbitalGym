# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""PE-6: evader forecast. The MPPI evader's forecast of the pursuer by its sample count.

The evader forecasts direct pursuit (MPPI-Direct) or LQR pursuit (MPPI-LQR) and
draws 64, 256, or 1,024 samples, at equal budgets and three thrust ratios.

Runs:
    pe_6  two forecasts by three sample counts by three thrust ratios against
          four pursuers (72 configurations)

Run on a GPU from the repository root (about 2 min on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/pe_6_evader_forecast.py

Name one or more runs to execute only those, for example ``pe_6_evader_forecast.py pe_6``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools

from baseline import pe_game
from pe_game import Planner
from runner import Run, main


def grid():
    """Evader forecast, thrust ratio, and evader sample count against four pursuers."""
    return [
        (pe_game(thrust, 1.0, pursuer=p, evader=e), Planner(evader_samples=samples))
        for p, e, thrust, samples in itertools.product(
            ("mppi_coast", "mppi_flee", "lqr", "hcw"),
            ("mppi_direct", "mppi_lqr"),
            (0.5, 0.75, 1.0),
            (64, 256, 1024),
        )
    ]


RUNS = (Run("pe_6", grid),)


if __name__ == "__main__":
    main(RUNS, __doc__)
