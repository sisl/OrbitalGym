"""End-to-end: 1 guard, 3 bandits — verify belief masking by range.

Two bandits placed within sensor range, one outside. After one belief update
step:
  - In-range bandit blocks have shrunk covariance (correction applied).
  - Out-of-range bandit block has only-predicted covariance (initial + Q).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from orbitalgym.belief.kf import KFBeliefUpdater, KFFromTruthInitializer
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import Actions, BySide, Side
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.observations.range_limited import RangeLimitedObservation


class _LayoutAdapter:
    def __init__(self, n_guards, n_bandits, d):
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


def test_three_bandits_two_in_range_one_out_belief_propagates_correctly():
    n_guards = 1
    n_bandits = 3
    d = 6
    sensor_range_m = 1500.0  # 1.5 km

    layout = _LayoutAdapter(n_guards, n_bandits, d)
    obs_fn = RangeLimitedObservation(layout=layout, sensor_range_m=sensor_range_m, sigma_range=10.0)

    cfg = make_lady_bandit_guard(n_guards=n_guards, n_bandits=n_bandits)
    cfg = dataclasses.replace(cfg, guard_observation_fn=obs_fn)
    env = OrbitalGymEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))

    # Manually overwrite bandit positions to known distances.
    bandits_rtn = state.bandits.rtn
    bandits_rtn = bandits_rtn.at[0, :3].set(jnp.array([300.0, 0.0, 0.0]))  # in range
    bandits_rtn = bandits_rtn.at[1, :3].set(jnp.array([800.0, 0.0, 0.0]))  # in range
    bandits_rtn = bandits_rtn.at[2, :3].set(jnp.array([5000.0, 0.0, 0.0]))  # out of range
    new_bandits = state.bandits.replace(rtn=bandits_rtn)
    # Place guard at origin so distance == bandit position magnitude.
    guards_rtn = state.guards.rtn.at[0, :].set(jnp.zeros(d))
    new_guards = state.guards.replace(rtn=guards_rtn)
    state = state.replace(guards=new_guards, bandits=new_bandits)

    init = KFFromTruthInitializer(layout=layout, variance_diag=jnp.ones(d) * 10.0)
    Q = jnp.eye(d) * 0.5  # noqa: N806
    upd = KFBeliefUpdater(
        stm=jnp.eye(d),
        control_matrix=jnp.zeros((d, 3)),
        process_noise=Q,
    )
    belief = init(state, Side.GUARD, jax.random.PRNGKey(1))
    initial_cov = belief.cov  # (1, 4, 6, 6)

    identity_actions = Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(env.config.n_guards),
            bandit=env.bandit_command_cls.zeros(env.config.n_bandits),
        )
    )
    obs_channels = obs_fn(state, identity_actions, Side.GUARD, cfg, jax.random.PRNGKey(2), state.t)
    new_belief = upd(
        belief, obs_channels, jnp.zeros((n_guards, 3)), Side.GUARD, jax.random.PRNGKey(3)
    )

    # Indices in (1, N_total=4): 0 = guard self, 1 = bandit 0, 2 = bandit 1, 3 = bandit 2.
    in_range_cov_a = new_belief.cov[0, 1]
    in_range_cov_b = new_belief.cov[0, 2]
    out_range_cov = new_belief.cov[0, 3]
    self_cov = new_belief.cov[0, 0]

    # Out-of-range bandit cov should equal predict-only (initial + Q).
    predict_only = initial_cov[0, 3] + Q
    assert jnp.allclose(out_range_cov, predict_only, atol=1e-6), (
        "Out-of-range bandit cov should equal predict-only (initial + Q)."
    )
    # In-range bandit position-block trace should drop below predict-only position-block trace.
    pos_slice = slice(0, 3)
    trace_in_a = float(jnp.trace(in_range_cov_a[pos_slice, pos_slice]))
    trace_in_b = float(jnp.trace(in_range_cov_b[pos_slice, pos_slice]))
    trace_out = float(jnp.trace(out_range_cov[pos_slice, pos_slice]))
    assert trace_in_a < trace_out, (
        f"in-range A trace {trace_in_a} should be < out trace {trace_out}"
    )
    assert trace_in_b < trace_out, (
        f"in-range B trace {trace_in_b} should be < out trace {trace_out}"
    )

    # Self pair under range-limited only: not corrected (visible[i, i] = False since
    # opposing_mask requires k >= N_self). Predict-only.
    assert jnp.allclose(self_cov, predict_only, atol=1e-6)
