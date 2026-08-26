"""LeadInterceptPursuer — generalized lead-intercept thrust toward
the opponent's predicted next-step position."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from orbitalgym import OrbitalGymEnv, SingleAgentView, make_pursuit_evasion
from orbitalgym.policies import ZeroControl
from orbitalgym.policies.heuristic import LeadInterceptPursuer
from orbitalgym.rollout import rollout_single_agent
from tests.policies._helpers import make_impulsive_maneuver_command_cls


def test_lead_intercept_emits_unit_thrust_along_predicted_line():
    p = LeadInterceptPursuer(
        max_dv_mps=0.05, dt=10.0, n_vehicles=1, command_cls=make_impulsive_maneuver_command_cls(1)
    )
    # FullObservation flattens [own_truth(6), opp_truth(6)] for 1v1 RTN.
    own = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Opponent at (100, 0, 0) with zero velocity → predicted next-step at (100, 0, 0).
    opp = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obs = jnp.concatenate([own, opp])
    cmd, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    # Unit direction +x, magnitude 0.05.
    assert cmd.dv.shape == (1, 3)
    assert pytest.approx(float(cmd.dv[0, 0]), abs=1e-6) == 0.05
    assert pytest.approx(float(cmd.dv[0, 1]), abs=1e-6) == 0.0
    assert pytest.approx(float(cmd.dv[0, 2]), abs=1e-6) == 0.0


def test_lead_intercept_runs_in_pe_rollout():
    cfg = make_pursuit_evasion(seed=0, max_horizon_s=200.0)
    cfg = dataclasses.replace(cfg, bandit_policy=LeadInterceptPursuer(max_dv_mps=0.05, dt=cfg.dt))
    env = OrbitalGymEnv(cfg)
    view = SingleAgentView(env)
    guard = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
    traj = rollout_single_agent(
        view, guard, lambda c, s, k: None, jax.random.PRNGKey(0), n_steps=cfg.max_steps
    )
    # Sanity: capture or run-out, not a crash.
    assert traj.episode_done.shape == (cfg.max_steps,)
