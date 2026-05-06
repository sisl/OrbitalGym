"""Customize rewards — source-of-truth for snippets in
docs/extending/customize-rewards.md.
"""

from __future__ import annotations

import jax


def test_customize_rewards_walkthrough():
    # --8<-- [start:imports]
    import dataclasses
    from dataclasses import dataclass

    import jax.numpy as jnp

    from orbital_game import OrbitalGameEnv, Side, SingleAgentView, make_lady_bandit_guard
    from orbital_game.policies import ZeroControl
    from orbital_game.rewards.base import RewardScope
    # --8<-- [end:imports]

    # --8<-- [start:sparse-breach-reward]
    @dataclass(frozen=True)
    class SparseBreachReward:
        """+0 every step; -1 to the guard on the step where the bandit
        breaches the lady (and +1 to the bandit, mirrored)."""

        breach_radius_m: float
        scope: RewardScope = RewardScope.PER_SIDE

        def __call__(self, prev_state, action, next_state, side, params, t):
            del prev_state, action, params, t
            bandits = next_state.bandits
            positions = bandits.rtn[:, :3] if hasattr(bandits, "rtn") else bandits.rt[:, :2]
            min_dist = jnp.min(jnp.linalg.norm(positions, axis=-1))
            breached = min_dist < self.breach_radius_m
            # Guard loses 1 on breach; bandit mirrors.
            penalty = jnp.where(breached, -1.0, 0.0)
            return jnp.where(side == Side.GUARD, penalty, -penalty)

    # --8<-- [end:sparse-breach-reward]

    # --8<-- [start:wire-it-up]
    cfg = make_lady_bandit_guard(seed=0, max_horizon_s=200.0)
    cfg = dataclasses.replace(cfg, reward_fn=SparseBreachReward(breach_radius_m=5.0))
    env = OrbitalGameEnv(cfg)
    # --8<-- [end:wire-it-up]

    view = SingleAgentView(env)
    guard = ZeroControl(command_cls=env.guard_command_cls, n_vehicles=cfg.n_guards)
    del guard  # smoke check; we drive zero actions directly below
    # Smoke check: env constructs and steps without crashing.
    state, obs, opp_ps = view.reset(jax.random.PRNGKey(0))
    del obs
    controlled_cmd = env.guard_command_cls.zeros(cfg.n_guards)
    next_state, next_obs, reward, done, next_opp_ps, info = view.step(
        jax.random.PRNGKey(1), state, controlled_cmd, opp_ps
    )
    del next_state, next_obs, done, next_opp_ps, info
    assert reward.shape == ()
