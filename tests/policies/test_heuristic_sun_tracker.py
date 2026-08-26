"""SunTrackerBlocker — maintain Sun-line geometry between opponent and Sun.

If the cfg's Sun direction is a fixed knob: the test injects a known Sun
unit vector and asserts the action lies along the projection of the
opponent-to-Sun line in the local frame.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbitalgym.policies.heuristic import SunTrackerBlocker
from tests.policies._helpers import make_impulsive_maneuver_command_cls


def test_sun_tracker_thrusts_toward_opponent_sun_line():
    # Sun direction +x (RTN); opponent at +y; expected: thrust toward x-axis projection.
    sun_dir_rtn = jnp.array([1.0, 0.0, 0.0])
    p = SunTrackerBlocker(
        sun_dir_rtn=sun_dir_rtn,
        max_dv_mps=0.05,
        n_vehicles=1,
        command_cls=make_impulsive_maneuver_command_cls(1),
    )
    own = jnp.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    opp = jnp.array([0.0, 200.0, 0.0, 0.0, 0.0, 0.0])
    obs = jnp.concatenate([own, opp])
    cmd, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    # Action should have a positive +x component (toward the Sun-side of the opponent).
    assert cmd.dv.shape == (1, 3)
    assert float(cmd.dv[0, 0]) > 0.0
    assert pytest.approx(float(jnp.linalg.norm(cmd.dv[0])), abs=1e-6) == 0.05
