"""Phase 0 regression test — proves the rename buckets do not change numerical
output. Loads the pre-rename reference-scenario trajectory (captured by
tests/fixtures/_generate_phase_0_baseline.py) and re-runs the scenario,
asserting array_equal on the unchanged leaves. The timeout reward was
intentionally corrected after this historical fixture was captured.

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
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide
from orbitalgym.policies import ZeroControl
from orbitalgym.rollout import rollout

FIXTURE = Path(__file__).parent.parent / "fixtures" / "phase_0_baseline.npz"


@pytest.fixture(scope="module")
def baseline():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def fresh_traj():
    cfg = build_config()
    env = OrbitalGymEnv(cfg)

    guard_policy = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
    bandit_policy = ZeroControl(n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls)

    def init_none(c, s, k):
        return None

    return rollout(
        env,
        BySide(guard=guard_policy, bandit=bandit_policy),
        BySide(guard=init_none, bandit=init_none),
        jax.random.PRNGKey(cfg.seed),
        n_steps=cfg.max_steps,
    )


def test_reward_prefix_identical_and_timeout_absorbing(baseline, fresh_traj):
    reward = np.asarray(fresh_traj.sides.guard.reward)
    np.testing.assert_array_equal(baseline["reward"][:-1], reward[:-1])
    cfg = build_config()
    separation = np.linalg.norm(
        np.asarray(fresh_traj.env_state.guards.rtn[-1, 0, :3])
        - np.asarray(fresh_traj.env_state.bandits.rtn[-1, 0, :3])
    )
    # This coasting fixture has no event or resource/separation penalty.
    # Phi(next)=0 at timeout, so its last reward is -gain*Phi(previous).
    np.testing.assert_allclose(
        reward[-1],
        cfg.reward_fn.shaping_gain * separation / cfg.reward_fn.shaping_scale_m,
        rtol=1e-12,
    )
    assert bool(fresh_traj.episode_done[-1])


def test_done_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["done"], np.asarray(fresh_traj.episode_done))


def test_action_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["action"], np.asarray(fresh_traj.sides.guard.action.dv))


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
