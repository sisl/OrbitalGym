"""Tests for derived-frame view materialization in env.step (Task 6.3).

Mixed-frame scenarios (truth=ECI, belief=RTN) must populate the non-truth
per-side state fields after each step so downstream readers can consume them
directly. Conversion uses the *propagated* reference orbit so that derived
views are temporally consistent with the truth state at t+dt.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def test_derived_rtn_view_populated_for_eci_truth():
    from orbital_game.env.core import OrbitalGameEnv
    from orbital_game.env.types import Actions, BySide
    from orbital_game.frames.conversions import eci_to_rtn
    from orbital_game.registry import DynamicsKey, StateComponentKey
    from tests.test_config_resolved_dynamics import _minimal_cfg

    cfg = _minimal_cfg(
        truth_dynamics=DynamicsKey.KEPLERIAN_ECI,
        policy_dynamics=DynamicsKey.HCW_RTN,
        guard_components=(StateComponentKey.ECI,),
        bandit_components=(StateComponentKey.ECI,),
    )
    env = OrbitalGameEnv(cfg)
    state, _ = env.reset(jax.random.PRNGKey(0))
    actions = Actions(
        sides=BySide(
            guard=jnp.zeros((cfg.n_guards, 3)),
            bandit=jnp.zeros((cfg.n_bandits, 3)),
        )
    )
    out = env.step(jax.random.PRNGKey(1), state, actions)
    # ECI is canonical (truth); RTN is derived.
    assert hasattr(out.state.guards, "eci")
    assert hasattr(out.state.guards, "rtn")
    assert out.state.guards.eci.shape == (cfg.n_guards, 6)
    assert out.state.guards.rtn.shape == (cfg.n_guards, 6)
    # The derived RTN view must be consistent with the truth ECI via eci_to_rtn.
    ref6 = jnp.concatenate(
        [
            out.state.reference_orbit.position_eci,
            out.state.reference_orbit.velocity_eci,
        ]
    )
    expected_rtn = eci_to_rtn(out.state.guards.eci, ref6)
    np.testing.assert_allclose(out.state.guards.rtn, expected_rtn, atol=1e-6)
