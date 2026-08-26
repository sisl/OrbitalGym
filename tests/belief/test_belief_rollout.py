"""Tests for ``orbitalgym.rollout.belief_rollout``.

The function:

- Threads per-side beliefs through a ``lax.scan``-based env loop.
- **Passes the post-update belief to each policy as ``agent_view``** —
  *not* the flattened obs. This is the load-bearing wart fix from the
  protocol unification: belief is the agent's perception output.
- Falls back to the same impulsive-maneuver fail-fast guard the old
  ``BeliefRollout`` class enforced.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from orbitalgym.belief.kf import KFBeliefUpdater, KFFromTruthInitializer
from orbitalgym.env.core import OrbitalGymEnv
from orbitalgym.env.types import BySide
from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard
from orbitalgym.observations.range_limited import RangeLimitedObservation
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.registry import ActionComponentKey
from orbitalgym.rollout import belief_rollout


class _LayoutAdapter:
    def __init__(self, n_guards, n_bandits, d):
        self.n_guards = n_guards
        self.n_bandits = n_bandits
        self.dynamics_state_dim = d


def _replace_obs_fn(cfg, obs_fn):
    return dataclasses.replace(cfg, guard_observation_fn=obs_fn, bandit_observation_fn=obs_fn)


def _make_env(cfg):
    return OrbitalGymEnv(cfg)


def _zero_init(_cfg, _state, _key):
    return None


@dataclass(frozen=True)
class _BeliefAssertingPolicy:
    """Policy that uses ``agent_view.mean`` as an array (not a method).

    If the rollout ever passed the flat obs (a ``jax.Array``) instead of the
    Belief, ``agent_view.mean`` would resolve to a bound method and
    ``jnp.sum(method)`` would raise — exactly the regression this test is
    guarding against.
    """

    n_vehicles: int = 0
    command_cls: Any = None

    def __call__(self, policy_state, agent_view, key, t):
        del key, t
        # Force `mean` to be used as an array. Using the value (not just
        # accessing it) prevents tracer-elimination from optimising it away.
        mean_sum = jnp.sum(agent_view.mean)
        cmd = self.command_cls.zeros(self.n_vehicles)
        # Embed the trace dependence so it cannot be DCE'd.
        cmd = cmd.replace(dv=cmd.dv + 0.0 * mean_sum)
        return cmd, policy_state


def test_belief_rollout_runs_one_step_and_returns_belief_history():
    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=2)
    layout = _LayoutAdapter(cfg.n_guards, cfg.n_bandits, 6)
    obs_fn = RangeLimitedObservation(layout=layout, sensor_range_m=1e9, sigma_range=1.0)
    cfg_with = _replace_obs_fn(cfg, obs_fn)
    env = _make_env(cfg_with)
    init = KFFromTruthInitializer(layout=layout, variance_diag=jnp.ones(6) * 5.0)
    upd = KFBeliefUpdater(
        stm=jnp.eye(6),
        control_matrix=jnp.zeros((6, 3)),
        process_noise=jnp.eye(6) * 0.01,
    )

    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits),
    )
    init_fns = BySide(guard=_zero_init, bandit=_zero_init)
    bel_inits = BySide(guard=init, bandit=init)
    bel_updates = BySide(guard=upd, bandit=upd)

    traj, history = belief_rollout(
        env,
        policies,
        init_fns,
        bel_inits,
        bel_updates,
        key=jax.random.PRNGKey(0),
        n_steps=1,
    )
    assert history.guard.mean.shape == (1, cfg.n_guards, cfg.n_guards + cfg.n_bandits, 6)
    assert history.bandit.mean.shape == (1, cfg.n_bandits, cfg.n_guards + cfg.n_bandits, 6)


def test_belief_rollout_passes_belief_to_policy_as_agent_view():
    """Core contract: policy.agent_view is the BELIEF, not the obs."""
    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1)
    layout = _LayoutAdapter(cfg.n_guards, cfg.n_bandits, 6)
    obs_fn = RangeLimitedObservation(layout=layout, sensor_range_m=1e9, sigma_range=1.0)
    cfg_with = _replace_obs_fn(cfg, obs_fn)
    env = _make_env(cfg_with)
    init = KFFromTruthInitializer(layout=layout, variance_diag=jnp.ones(6))
    upd = KFBeliefUpdater(
        stm=jnp.eye(6),
        control_matrix=jnp.zeros((6, 3)),
        process_noise=jnp.eye(6) * 0.01,
    )

    rec_g = _BeliefAssertingPolicy(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
    rec_b = _BeliefAssertingPolicy(n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls)
    policies = BySide(guard=rec_g, bandit=rec_b)
    init_fns = BySide(guard=_zero_init, bandit=_zero_init)
    bel_inits = BySide(guard=init, bandit=init)
    bel_updates = BySide(guard=upd, bandit=upd)

    # Smoke run — if `agent_view` were the obs (a 1-D array), `agent_view.mean`
    # access in `_BeliefRecordingPolicy` would resolve to the JAX array's
    # `.mean` method (a callable), and the tree_map below would receive a
    # `MethodWrapper`; the trace would fail. With the belief contract, mean
    # is a real array attribute and the trace succeeds.
    traj, history = belief_rollout(
        env,
        policies,
        init_fns,
        bel_inits,
        bel_updates,
        key=jax.random.PRNGKey(0),
        n_steps=1,
    )
    assert history.guard.mean.shape[0] == 1


def test_belief_rollout_observation_shrinks_belief_covariance():
    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=2)
    layout = _LayoutAdapter(cfg.n_guards, cfg.n_bandits, 6)
    obs_fn = RangeLimitedObservation(layout=layout, sensor_range_m=1e9, sigma_range=1.0)
    cfg_with = _replace_obs_fn(cfg, obs_fn)
    env = _make_env(cfg_with)
    init = KFFromTruthInitializer(layout=layout, variance_diag=jnp.ones(6) * 5.0)
    upd = KFBeliefUpdater(
        stm=jnp.eye(6),
        control_matrix=jnp.zeros((6, 3)),
        process_noise=jnp.eye(6) * 0.01,
    )
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits),
    )
    init_fns = BySide(guard=_zero_init, bandit=_zero_init)
    bel_inits = BySide(guard=init, bandit=init)
    bel_updates = BySide(guard=upd, bandit=upd)

    _, history = belief_rollout(
        env,
        policies,
        init_fns,
        bel_inits,
        bel_updates,
        key=jax.random.PRNGKey(0),
        n_steps=2,
    )
    # After-step variance trace should be smaller than initial-only variance
    # trace (the prior in the initializer is 5.0; KF correct shrinks it).
    cov_t1 = float(jnp.trace(history.guard.cov[1, 0, 1]))
    initial_trace = 6.0 * 5.0  # diag init
    assert cov_t1 < initial_trace


def test_belief_rollout_without_impulsive_maneuver_raises():
    cfg = make_lady_bandit_guard(n_guards=1, n_bandits=1)
    cfg_no_dv = dataclasses.replace(
        cfg,
        guard_action_components=(ActionComponentKey.COMMUNICATE,),
        bandit_action_components=(ActionComponentKey.COMMUNICATE,),
    )
    env = _make_env(cfg_no_dv)
    init = KFFromTruthInitializer(
        layout=_LayoutAdapter(cfg.n_guards, cfg.n_bandits, 6),
        variance_diag=jnp.ones(6),
    )
    upd = KFBeliefUpdater(
        stm=jnp.eye(6),
        control_matrix=jnp.zeros((6, 3)),
        process_noise=jnp.eye(6) * 0.01,
    )
    policies = BySide(
        guard=ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards),
        bandit=ZeroControl(command_cls=env.bandit_command_cls, n_vehicles=cfg.n_bandits),
    )
    init_fns = BySide(guard=_zero_init, bandit=_zero_init)

    with pytest.raises(ValueError, match="IMPULSIVE_MANEUVER"):
        belief_rollout(
            env,
            policies,
            init_fns,
            BySide(guard=init, bandit=init),
            BySide(guard=upd, bandit=upd),
            key=jax.random.PRNGKey(0),
            n_steps=1,
        )
