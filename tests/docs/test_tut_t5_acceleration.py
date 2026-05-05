"""T5 — GPU / MPS acceleration. Source-of-truth for snippets in
docs/tutorials/t5-acceleration.md.

CI runs this on CPU; the test exercises the vmap-over-seeds shape that
gives the meaningful speedup on GPU.
"""

from __future__ import annotations


def test_t5_vmap_over_seeds():
    # --8<-- [start:imports]
    import jax
    import jax.numpy as jnp

    from orbital_game import (
        OrbitalGameEnv,
        SingleAgentView,
        make_pursuit_evasion,
    )
    from orbital_game.policies import ZeroControl
    from orbital_game.rollout import rollout_single_agent
    # --8<-- [end:imports]

    # --8<-- [start:devices]
    print(jax.devices())  # [CudaDevice(id=0)] on CUDA, [CpuDevice(id=0)] on CPU
    # --8<-- [end:devices]

    cfg = make_pursuit_evasion(seed=0, max_horizon_s=1000.0)
    env = OrbitalGameEnv(cfg)
    view = SingleAgentView(env)
    guard_policy = ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards)

    # --8<-- [start:vmap-rollout]
    def run_one(seed: jax.Array):
        return rollout_single_agent(
            view,
            guard_policy,
            lambda c, s, k: None,
            seed,
            n_steps=cfg.max_steps,
        )

    n_seeds = 32  # 1024 in real use; small here for CI speed
    seeds = jax.vmap(jax.random.PRNGKey)(jnp.arange(n_seeds))
    trajs = jax.jit(jax.vmap(run_one))(seeds)
    # --8<-- [end:vmap-rollout]

    # --8<-- [start:read-batched]
    # Leading axis is now the seed batch:
    rewards = trajs.sides.guard.reward  # (B, T)
    final_rewards = rewards.sum(axis=-1)  # (B,)
    mean_return = float(final_rewards.mean())
    # --8<-- [end:read-batched]

    assert rewards.shape[0] == n_seeds
    assert rewards.shape[1] == cfg.max_steps
    assert isinstance(mean_return, float)
