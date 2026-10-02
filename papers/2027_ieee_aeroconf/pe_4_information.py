# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""PE-4: information. Sensor geometry, range, and noise in place of full information.

With a sensor, both sides estimate the relative state with a Kalman filter from
range-proportional position measurements and start from a Gaussian prior error.
A cone sensor points along the estimated line of sight.

Runs:
    pe_4             six sensors by two separations for every pursuer against
                     four evaders at three operating points (1,008
                     configurations), for Figure 6
    pe_4_capability  the capability sweep of PE-1 with a 10 degree, 5 km cone
                     sensor and 1% noise on both sides (3,780 configurations)

Run on a GPU from the repository root (about 20 min on one NVIDIA H100):

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/pe_4_information.py

Name one or more runs to execute only those, for example ``pe_4_information.py pe_4``.
Records go to ``results/<run>/`` and summaries to ``summary/<run>.csv``, both
next to this script.
"""

import itertools

from baseline import FOCUS_EVADERS, OPERATING_POINTS, PURSUERS, capability_grid, pe_game
from runner import Run, main

SENSORS = {
    "full": {},
    "radial_far": dict(sensor="radial", sensor_range_m=5000.0),
    "radial_near": dict(sensor="radial", sensor_range_m=500.0),
    "cone_far": dict(sensor="cone", sensor_range_m=5000.0, cone_half_angle_deg=10.0),
    "cone_near": dict(sensor="cone", sensor_range_m=500.0, cone_half_angle_deg=10.0),
    "radial_far_noisy": dict(sensor="radial", sensor_range_m=5000.0, noise_fraction=0.1),
}


def grid():
    """Sensor by separation; the prior position error is one tenth of the separation."""
    games = []
    for name, (thrust, fuel), p, e, d in itertools.product(
        SENSORS, OPERATING_POINTS, PURSUERS, FOCUS_EVADERS, (300.0, 1000.0)
    ):
        sensing = SENSORS[name]
        if sensing:
            sensing = dict(sensing, prior_sigma_m=0.1 * d, prior_sigma_mps=0.01)
        games.append(pe_game(thrust, fuel, pursuer=p, evader=e, separation_m=d, **sensing))
    return games


def capability_under_cone_sensor():
    """The capability sweep of PE-1 with the far cone sensor on both sides."""
    return capability_grid(**SENSORS["cone_far"], prior_sigma_m=30.0, prior_sigma_mps=0.01)


RUNS = (Run("pe_4", grid), Run("pe_4_capability", capability_under_cone_sensor))


if __name__ == "__main__":
    main(RUNS, __doc__)
