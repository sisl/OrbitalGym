"""In-depth: Symmetric core & data shapes. Source-of-truth for snippets in
docs/in-depth/symmetric-core.md.
"""

from __future__ import annotations


def test_indepth_symmetric_core():
    # --8<-- [start:byside]
    import jax
    import jax.numpy as jnp

    from orbital_game import BySide, Side

    bs = BySide(guard=42.0, bandit=-1.0)
    assert bs.get(Side.GUARD) == 42.0
    assert bs.map(lambda x: x * 2).bandit == -2.0
    # --8<-- [end:byside]

    # --8<-- [start:byside-pytree]
    # BySide is a JAX pytree — vmap and tree.map traverse it.
    bs_arr = BySide(guard=jnp.array([1.0, 2.0]), bandit=jnp.array([3.0, 4.0]))
    bs_squared = jax.tree.map(lambda x: x**2, bs_arr)
    assert bool(jnp.all(bs_squared.bandit == jnp.array([9.0, 16.0])))
    # --8<-- [end:byside-pytree]

    # --8<-- [start:rollout-shapes]
    from orbital_game import OrbitalGameEnv, SingleAgentView, make_pursuit_evasion
    from orbital_game.policies import ZeroControl
    from orbital_game.rollout import rollout_single_agent

    cfg = make_pursuit_evasion(n_guards=2, n_bandits=1, seed=0, max_horizon_s=1000.0)
    env = OrbitalGameEnv(cfg)
    view = SingleAgentView(env)
    guard_policy = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
    traj = rollout_single_agent(
        view,
        guard_policy,
        lambda c, s, k: None,
        jax.random.PRNGKey(0),
        n_steps=cfg.max_steps,
    )

    # T=time, N=vehicle, feature
    assert traj.sides.guard.action.shape == (cfg.max_steps, cfg.n_guards, 3)
    assert traj.sides.guard.reward.shape == (cfg.max_steps,)
    assert traj.env_state.guards.rtn.shape == (cfg.max_steps, cfg.n_guards, 6)
    # --8<-- [end:rollout-shapes]

    # --8<-- [start:indexing-recipes]
    # i-th guard's reward time series — well, reward is PER_SIDE for
    # the bundled scope, so it's just `traj.sides.guard.reward`. For
    # PER_VEHICLE rewards it would be `traj.sides.guard.reward[:, i]`.
    side_reward_t = traj.sides.guard.reward  # (T,)

    # i-th guard's RTN trajectory:
    guard0_rtn = traj.env_state.guards.rtn[:, 0]  # (T, 6)

    # Step t's full state across all guards:
    step5_state = traj.env_state.guards.rtn[5]  # (N_g, 6)
    # --8<-- [end:indexing-recipes]

    # --8<-- [start:episode-mask]
    from orbital_game.rollout import episode_mask

    valid = episode_mask(traj)  # (T,) bool, latched
    valid_rewards = side_reward_t[valid]  # only steps before termination
    # --8<-- [end:episode-mask]

    # --8<-- [start:vmap-shape]
    # Under vmap over seeds, every shape gains a leading B dimension:
    def run_one(seed):
        return rollout_single_agent(
            view,
            guard_policy,
            lambda c, s, k: None,
            seed,
            n_steps=cfg.max_steps,
        )

    seeds = jax.vmap(jax.random.PRNGKey)(jnp.arange(4))
    batched = jax.jit(jax.vmap(run_one))(seeds)
    assert batched.sides.guard.action.shape == (4, cfg.max_steps, cfg.n_guards, 3)
    # B=batch, T=time, N=vehicle, feature
    # --8<-- [end:vmap-shape]

    assert valid_rewards.shape[0] >= 1
    assert step5_state.shape == (cfg.n_guards, 6)
    assert guard0_rtn.shape == (cfg.max_steps, 6)
