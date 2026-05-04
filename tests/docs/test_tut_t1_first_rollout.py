"""T1 — First rollout (RT 2D). Source-of-truth for snippets in
docs/tutorials/t1-first-rollout.md.
"""

from __future__ import annotations


def test_t1_first_rollout_rt2d():
    # --8<-- [start:imports]
    import jax

    from orbital_game import OrbitalGameEnv, SingleAgentView, make_lady_bandit_guard
    from orbital_game.policies import ZeroControl
    from orbital_game.registry import DynamicsKey, StateComponentKey
    from orbital_game.rollout import rollout_single_agent
    # --8<-- [end:imports]

    # --8<-- [start:build-config]
    cfg = make_lady_bandit_guard(
        n_guards=1,
        n_bandits=1,
        truth_dynamics=DynamicsKey.HCW_RT,
        planning_dynamics=DynamicsKey.HCW_RT,
        guard_components=(StateComponentKey.RT, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RT,),
        max_horizon_s=2000.0,
        seed=0,
    )
    # --8<-- [end:build-config]

    # --8<-- [start:run-rollout]
    env = OrbitalGameEnv(cfg)
    view = SingleAgentView(env)

    guard_policy = ZeroControl(n_vehicles=cfg.n_guards, action_dim=2)
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

    assert total_reward < 0.0  # negative-distance reward — closed ellipse
    assert state_shape == (cfg.max_steps, cfg.n_guards, 4)
    assert n_steps_actual >= 1
