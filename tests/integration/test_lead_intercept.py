"""LeadInterceptPursuer ought to close on a non-evading guard.

The bandit should drive the relative distance below its IC distance over
the episode when the guard is on ZeroControl.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from examples.policies.lead_intercept import LeadInterceptPursuer
from orbital_game import (
    OrbitalGameEnv,
    SingleAgentView,
    make_pursuit_evasion,
)
from orbital_game.policies import ZeroControl
from orbital_game.rollout import rollout_single_agent


def _rollout_min_dist(bandit_policy):
    cfg = make_pursuit_evasion(seed=0, max_horizon_s=2000.0)
    cfg = dataclasses.replace(cfg, bandit_policy=bandit_policy)
    env = OrbitalGameEnv(cfg)
    view = SingleAgentView(env)
    guard_policy = ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards)

    traj = rollout_single_agent(
        view,
        guard_policy,
        lambda c, s, k: None,
        jax.random.PRNGKey(0),
        n_steps=cfg.max_steps,
    )

    guard_pos = traj.env_state.guards.rtn[..., :3]  # (T, N_g, 3)
    bandit_pos = traj.env_state.bandits.rtn[..., :3]  # (T, N_b, 3)
    dist = jnp.linalg.norm(guard_pos[:, 0, :] - bandit_pos[:, 0, :], axis=-1)
    return float(dist[0]), float(dist.min())


def test_lead_intercept_closes_distance_on_static_guard():
    cfg = make_pursuit_evasion(seed=0, max_horizon_s=2000.0)

    initial_dist, lead_min = _rollout_min_dist(LeadInterceptPursuer(max_dv_mps=0.05, dt=cfg.dt))
    _, zero_min = _rollout_min_dist(ZeroControl())

    # Closest approach over the episode rather than the final-step distance:
    # HCW dynamics over a multi-orbit horizon naturally swing the relative
    # trajectory past closest approach, so a final-step assertion is noisy
    # even when the policy is effective.
    assert lead_min < initial_dist, (
        f"LeadIntercept failed to close: initial={initial_dist:.1f}, min={lead_min:.1f}"
    )
    # Guard against passing by accident — IC dynamics alone close some
    # distance, so require the active policy to do meaningfully better
    # than a passive bandit (ZeroControl-vs-ZeroControl).
    assert lead_min < 0.75 * zero_min, (
        f"LeadIntercept did not improve over ZeroControl: "
        f"lead_min={lead_min:.1f}, zero_min={zero_min:.1f}"
    )
