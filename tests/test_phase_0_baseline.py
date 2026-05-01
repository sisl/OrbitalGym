"""Phase 0 regression test — proves the rename buckets do not change numerical
output. Loads the pre-rename reference-scenario trajectory (captured by
tests/fixtures/_generate_phase_0_baseline.py) and re-runs the scenario,
asserting array_equal on every leaf.

Function names and fixture keys use post-rename clean names from Day 0.
Attribute accesses on `fresh_traj` track current code state and update
bucket-by-bucket:
  - Bucket A: DONE (env_state.hva → env_state.reference_orbit)
  - Bucket B: DONE (env_state.defenders → env_state.guards)
  - Bucket C: DONE (env_state.intruders → env_state.bandits)

The fixture file itself is NEVER regenerated during Phase 0 — that's the
load-bearing byte-identity guarantee.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.reference_scenario import build_config
from orbital_game.env.environment import OrbitalGameEnv
from orbital_game.rollout import rollout

FIXTURE = Path(__file__).parent / "fixtures" / "phase_0_baseline.npz"


@pytest.fixture(scope="module")
def baseline():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def fresh_traj():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)

    def policy(ps, obs, key, t):
        del obs, key, t
        # Bucket B: DONE
        return jnp.zeros((cfg.n_guards, 3)), ps

    def init_ps(config, env_state, key):
        del config, env_state, key
        return None

    return rollout(env, policy, init_ps, jax.random.PRNGKey(cfg.seed), n_steps=cfg.max_steps)


def test_reward_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["reward"], np.asarray(fresh_traj.reward))


def test_done_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["done"], np.asarray(fresh_traj.done))


def test_action_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["action"], np.asarray(fresh_traj.action))


def test_obs_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["obs"], np.asarray(fresh_traj.obs))


def test_guards_rtn_identical(baseline, fresh_traj):
    # Bucket B: DONE
    np.testing.assert_array_equal(
        baseline["guards_rtn"], np.asarray(fresh_traj.env_state.guards.rtn)
    )


def test_bandits_rtn_identical(baseline, fresh_traj):
    # Bucket C: DONE (env_state.intruders → env_state.bandits)
    np.testing.assert_array_equal(
        baseline["bandits_rtn"], np.asarray(fresh_traj.env_state.bandits.rtn)
    )


def test_guards_propellant_mass_identical(baseline, fresh_traj):
    # Bucket B: DONE
    np.testing.assert_array_equal(
        baseline["guards_propellant_mass"],
        np.asarray(fresh_traj.env_state.guards.propellant_mass),
    )


def test_reference_orbit_position_eci_identical(baseline, fresh_traj):
    # Bucket A: DONE
    np.testing.assert_array_equal(
        baseline["reference_orbit_position_eci"],
        np.asarray(fresh_traj.env_state.reference_orbit.position_eci),
    )


def test_reference_orbit_velocity_eci_identical(baseline, fresh_traj):
    # Bucket A: DONE
    np.testing.assert_array_equal(
        baseline["reference_orbit_velocity_eci"],
        np.asarray(fresh_traj.env_state.reference_orbit.velocity_eci),
    )


def test_t_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["t"], np.asarray(fresh_traj.env_state.t))


def test_step_identical(baseline, fresh_traj):
    np.testing.assert_array_equal(baseline["step"], np.asarray(fresh_traj.env_state.step))
