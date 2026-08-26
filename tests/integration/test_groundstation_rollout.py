"""Integration tests for `belief_rollout`'s contact-aware mode.

Verifies that the new optional kwargs (`guard_ground_station_network`,
`bandit_ground_station_network`, `team_sync_fns`) thread through the JAX
scan without breaking the existing belief-rollout contract.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbitalgym.belief.sync import KFTeamFusion
from orbitalgym.env.types import BySide
from orbitalgym.groundstations import (
    ContactSchedule,
    GroundStation,
    GroundStationNetwork,
)


def _network() -> GroundStationNetwork:
    station = GroundStation(
        name="A",
        lat_deg=jnp.asarray(0.0),
        lon_deg=jnp.asarray(0.0),
        altitude_m=jnp.asarray(0.0),
        elevation_mask_deg=jnp.asarray(5.0),
    )
    sch = ContactSchedule(
        windows=jnp.asarray(
            [[100.0, 300.0], [800.0, 1000.0]] + [[-1.0, -1.0]] * 6,
            dtype=jnp.float32,
        ),
        n_valid=jnp.asarray(2),
        station_ix=jnp.asarray([0, 0, -1, -1, -1, -1, -1, -1]),
    )
    return GroundStationNetwork(stations=(station,), schedule=sch)


def test_belief_rollout_accepts_network_kwargs(make_minimal_lbg_env_with_kf):
    """Smoke test: belief_rollout accepts the new kwargs without raising."""
    from orbitalgym.rollout import belief_rollout

    env, init_belief_fns, update_belief_fns, init_ps_fns, policies = make_minimal_lbg_env_with_kf()

    traj, beliefs = belief_rollout(
        env=env,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=init_belief_fns,
        belief_updaters=update_belief_fns,
        key=jax.random.key(0),
        n_steps=20,
        guard_ground_station_network=_network(),
        bandit_ground_station_network=None,
        team_sync_fns=None,
    )
    assert traj.env_state.t.shape == (20,)


def test_belief_rollout_default_kwargs_unchanged(make_minimal_lbg_env_with_kf):
    """Existing callers don't pass the new kwargs — must work unchanged."""
    from orbitalgym.rollout import belief_rollout

    env, init_belief_fns, update_belief_fns, init_ps_fns, policies = make_minimal_lbg_env_with_kf()

    traj, beliefs = belief_rollout(
        env=env,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=init_belief_fns,
        belief_updaters=update_belief_fns,
        key=jax.random.key(0),
        n_steps=10,
    )
    assert traj.env_state.t.shape == (10,)


def test_belief_rollout_applies_team_fusion(make_minimal_lbg_env_with_kf):
    env, init_belief_fns, update_belief_fns, init_ps_fns, policies = make_minimal_lbg_env_with_kf(
        n_guards=2,
    )

    from orbitalgym.rollout import belief_rollout

    sync_fns = BySide(guard=KFTeamFusion(), bandit=None)
    traj, beliefs = belief_rollout(
        env=env,
        policies=policies,
        init_policy_state_fns=init_ps_fns,
        belief_initializers=init_belief_fns,
        belief_updaters=update_belief_fns,
        key=jax.random.key(0),
        n_steps=120,
        guard_ground_station_network=_network(),
        bandit_ground_station_network=None,
        team_sync_fns=sync_fns,
    )
    # Pick a step inside the first contact window (start at t=100, dt=10 → step ~10).
    # During contact, both guards should have identical posteriors over the
    # bandit (k=2 in the (N_obs, N_total)=(2, 3) layout).
    cov_in_contact = beliefs.guard.cov[15]  # arbitrary in-contact step
    assert jnp.allclose(cov_in_contact[0, 2], cov_in_contact[1, 2], atol=1e-6)
