"""T1 — First rollout (RT 2D). Source-of-truth for snippets in
docs/tutorials/t1-first-rollout.md.
"""

from __future__ import annotations

import numpy as np


def test_t1_first_rollout_rt2d():
    # --8<-- [start:imports]
    import jax

    from orbitalgym import OrbitalGymEnv, SingleAgentView, make_lady_bandit_guard
    from orbitalgym.policies import ZeroControl
    from orbitalgym.registry import DynamicsKey, StateComponentKey
    from orbitalgym.rollout import rollout_single_agent
    # --8<-- [end:imports]

    # --8<-- [start:build-config]
    cfg = make_lady_bandit_guard(
        n_guards=1,
        n_bandits=1,
        truth_dynamics=DynamicsKey.HCW_RT,
        policy_dynamics=DynamicsKey.HCW_RT,
        guard_components=(StateComponentKey.RT, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RT,),
        max_horizon_s=2000.0,
        seed=0,
    )
    # --8<-- [end:build-config]

    # --8<-- [start:run-rollout]
    env = OrbitalGymEnv(cfg)
    view = SingleAgentView(env)

    guard_policy = ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards)
    traj = rollout_single_agent(
        view,
        guard_policy,
        lambda c, s, k: None,
        jax.random.PRNGKey(cfg.seed),
        n_steps=cfg.max_steps,
    )
    # --8<-- [end:run-rollout]

    # --8<-- [start:inspect]
    total_reward = float(traj.sides.guard.reward.sum())
    n_steps_actual = int(traj.episode_done.argmax()) + 1
    state_shape = traj.env_state.guards.rt.shape  # (T, N_g, 4)
    # --8<-- [end:inspect]

    # Undiscounted potential shaping telescopes to -gain * Phi(initial),
    # including the zero-potential timeout. It is not a distance sum.
    initial_separation = np.linalg.norm(
        np.asarray(traj.env_state.guards.rt[0, 0, :2])
        - np.asarray(traj.env_state.bandits.rt[0, 0, :2])
    )
    expected = cfg.reward_fn.shaping_gain * initial_separation / cfg.reward_fn.shaping_scale_m
    np.testing.assert_allclose(total_reward, expected, rtol=1e-6)
    assert bool(traj.episode_done[-1])
    assert state_shape == (cfg.max_steps, cfg.n_guards, 4)
    assert n_steps_actual >= 1
