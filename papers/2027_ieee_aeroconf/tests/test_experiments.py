"""Every experiment defines the configurations that the paper's tables state."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from controllers import controller_matrices, pursuit_command
from pe_game import REFERENCE_RADIUS_M

from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.policies.heuristic.feedback import hcw_lqr_gain, pd_impulse
from orbitalgym.reference_orbit import MU_EARTH

CONFIGURATIONS = {
    "pe_1_capability": {"pe_1": 3780, "pe_1_fine": 5766, "pe_1_seed2": 3780, "pe_1_n512": 3780},
    "pe_2_budget_size": {"pe_2": 7938, "pe_2_fine": 11532},
    "pe_3_geometry": {"pe_3": 3024, "pe_3_fine": 9672},
    "pe_4_information": {"pe_4": 1008, "pe_4_capability": 3780},
    "pe_5_lookahead": {"pe_5": 45},
    "pe_6_evader_forecast": {"pe_6": 72},
    "pe_7_capture_definition": {"pe_7": 864, "pe_7_endpoint": 3780},
    "pe_8_search": {"pe_8": 250, "pe_8_hcw_lookahead": 1080},
    "lbg_1_capability": {"lbg_1": 4410, "lbg_1_fine": 13454, "lbg_1_n512": 4410},
    "lbg_2_geometry": {"lbg_2": 784},
    "lbg_3_event_definition": {"lbg_3": 882},
}


@pytest.mark.parametrize("module", sorted(CONFIGURATIONS))
def test_runs_define_distinct_configurations(module):
    runs = {run.name: run for run in importlib.import_module(module).RUNS}
    assert set(runs) == set(CONFIGURATIONS[module])
    for name, count in CONFIGURATIONS[module].items():
        keys = [game.key(planner) for game, planner in runs[name].entries()]
        assert len(keys) == count
        assert len(set(keys)) == count


def test_pursuit_laws_use_the_package_feedback_policies():
    mean_motion = np.sqrt(MU_EARTH / REFERENCE_RADIUS_M**3)
    matrices = controller_matrices(mean_motion, 10.0, 100.0)
    np.testing.assert_array_equal(matrices[0], hcw_lqr_gain(mean_motion, 10.0))
    x = jnp.array([120.0, -80.0, 40.0, 0.3, -0.1, 0.2])
    key = jax.random.PRNGKey(0)
    np.testing.assert_allclose(pursuit_command(1, x, matrices, 0.1, key, 10.0), -matrices[0] @ x)
    np.testing.assert_allclose(pursuit_command(5, x, matrices, 0.1, key, 10.0), pd_impulse(x, 10.0))
    # The HCW intercept impulse brings the coasting relative position to zero at the lookahead.
    after = hcw_rtn_stm(mean_motion, 100.0) @ (
        x + jnp.concatenate([jnp.zeros(3), pursuit_command(2, x, matrices, 0.1, key, 10.0)])
    )
    np.testing.assert_allclose(after[:3], 0.0, atol=1e-6)
