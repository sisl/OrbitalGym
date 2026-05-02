"""Phase 0 regression test — proves the rename buckets do not change numerical
output. Loads the pre-rename reference-scenario trajectory (captured by
tests/fixtures/_generate_phase_0_baseline.py) and re-runs the scenario,
asserting array_equal on every leaf.

Phase 1 update: migrated from old single-agent rollout API to the new
BySide-of-policies rollout. ZeroControl on both sides preserves byte
identity with the pre-Phase-0 / pre-Phase-1 trajectory.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from examples.reference_scenario import build_config
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import BySide
from orbital_game.policies.library import ZeroControl
from orbital_game.rollout import rollout

FIXTURE = Path(__file__).parent.parent / "fixtures" / "phase_0_baseline.npz"


@pytest.fixture(scope="module")
def baseline():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def fresh_traj():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)

    guard_policy = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
    bandit_policy = ZeroControl(n_vehicles=cfg.n_bandits, action_dim=3)

    def init_none(c, s, k):
        return None

    return rollout(
        env,
        BySide(guard=guard_policy, bandit=bandit_policy),
        BySide(guard=init_none, bandit=init_none),
        jax.random.PRNGKey(cfg.seed),
        n_steps=cfg.max_steps,
    )


def test_reward_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["reward"], np.asarray(fresh_traj.sides.guard.reward))


def test_done_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["done"], np.asarray(fresh_traj.episode_done))


def test_action_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["action"], np.asarray(fresh_traj.sides.guard.action))


def test_obs_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["obs"], np.asarray(fresh_traj.sides.guard.obs))


def test_guards_rtn_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(
        baseline["guards_rtn"], np.asarray(fresh_traj.env_state.guards.rtn)
    )


def test_bandits_rtn_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(
        baseline["bandits_rtn"], np.asarray(fresh_traj.env_state.bandits.rtn)
    )


def test_guards_propellant_mass_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(
        baseline["guards_propellant_mass"],
        np.asarray(fresh_traj.env_state.guards.propellant_mass),
    )


def test_reference_orbit_position_eci_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(
        baseline["reference_orbit_position_eci"],
        np.asarray(fresh_traj.env_state.reference_orbit.position_eci),
    )


def test_reference_orbit_velocity_eci_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(
        baseline["reference_orbit_velocity_eci"],
        np.asarray(fresh_traj.env_state.reference_orbit.velocity_eci),
    )


def test_t_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["t"], np.asarray(fresh_traj.env_state.t))


def test_step_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["step"], np.asarray(fresh_traj.env_state.step))
