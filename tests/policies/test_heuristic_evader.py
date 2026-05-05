"""OrthogonalEvader — moves perpendicular to opponent relative velocity."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from orbital_game.policies.heuristic import OrthogonalEvader
from tests.policies._helpers import make_impulsive_maneuver_command_cls


def test_evader_action_is_orthogonal_to_relative_velocity():
    p = OrthogonalEvader(
        max_dv_mps=0.05, n_vehicles=1, command_cls=make_impulsive_maneuver_command_cls(1)
    )
    # Opponent approaches +x at 1 m/s relative.
    own = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    opp = jnp.array([100.0, 0.0, 0.0, -1.0, 0.0, 0.0])  # closing along -x
    obs = jnp.concatenate([own, opp])
    cmd, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    # Relative velocity (opp - own) is (-1, 0, 0). Orthogonal subspace is yz-plane.
    rel_v = jnp.array([-1.0, 0.0, 0.0])
    dot = float(jnp.dot(cmd.dv[0], rel_v))
    assert pytest.approx(dot, abs=1e-6) == 0.0
    assert pytest.approx(float(jnp.linalg.norm(cmd.dv[0])), abs=1e-6) == 0.05


def test_evader_handles_zero_relative_velocity_safely():
    p = OrthogonalEvader(
        max_dv_mps=0.05, n_vehicles=1, command_cls=make_impulsive_maneuver_command_cls(1)
    )
    own = jnp.zeros(6)
    opp = jnp.zeros(6)
    obs = jnp.concatenate([own, opp])
    cmd, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    # Defined fallback: emit a deterministic non-NaN unit thrust on a fixed axis.
    assert jnp.all(jnp.isfinite(cmd.dv))
    assert pytest.approx(float(jnp.linalg.norm(cmd.dv[0])), abs=1e-6) == 0.05
