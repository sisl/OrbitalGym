# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""LBG-3: event definition. Capture radius, breach radius, and a velocity-matching condition.

The guard has a thrust limit of 0.1 m/s per step and a 5 m/s delta-v budget. By
default the bandit starts 1,000 m and the guard 200 m from the lady, the game
lasts 1,800 s, the capture radius is 10 m, and the breach radius is 50 m.

Runs:
    lbg_3  three capture radii by three breach radii, with no velocity-matching
           condition or 0.5 m/s on both events, for every guard against every
           bandit at equal thrust limits and budgets (882 configurations)

Run on a GPU from the repository root (about 6 min on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/lbg_3_event_definition.py

Name one or more runs to execute only those, for example ``lbg_3_event_definition.py lbg_3``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools

import numpy as np
from baseline import lbg_game
from lbg_game import BANDITS, GUARDS
from runner import Run, main


def grid():
    """Capture radius, breach radius, and the velocity-matching condition on both events."""
    return [
        lbg_game(
            1.0,
            1.0,
            guard=g,
            bandit=b,
            capture_m=capture,
            breach_m=breach,
            capture_speed_mps=speed,
            breach_speed_mps=speed,
        )
        for g, b, capture, breach, speed in itertools.product(
            GUARDS, BANDITS, (5.0, 10.0, 20.0), (20.0, 50.0, 100.0), (np.inf, 0.5)
        )
    ]


RUNS = (Run("lbg_3", grid),)


if __name__ == "__main__":
    main(RUNS, __doc__)
