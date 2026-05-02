"""BeliefRollout — drives env.step + per-side belief updates.

Helper for tests, notebooks, ad-hoc scripts. Production planners are expected
to thread belief themselves; this is not load-bearing on the env loop.
"""

from __future__ import annotations

from typing import Any

import jax

from orbital_game.env.types import Actions, BySide, Side


class BeliefRollout:
    """Pairs an OrbitalGameEnv with per-side belief initializers + updaters."""

    def __init__(
        self,
        env: Any,
        guard_belief_initializer: Any,
        guard_belief_updater: Any,
        bandit_belief_initializer: Any,
        bandit_belief_updater: Any,
    ):
        self.env = env
        self.guard_belief_initializer = guard_belief_initializer
        self.guard_belief_updater = guard_belief_updater
        self.bandit_belief_initializer = bandit_belief_initializer
        self.bandit_belief_updater = bandit_belief_updater

    def reset(self, key: jax.Array):
        """Returns (env_state, BySide(guard=KFBelief, bandit=KFBelief))."""
        k_env, k_g_init, k_b_init = jax.random.split(key, 3)
        state, _outs = self.env.reset(k_env)
        beliefs = BySide(
            guard=self.guard_belief_initializer(state, Side.GUARD, k_g_init),
            bandit=self.bandit_belief_initializer(state, Side.BANDIT, k_b_init),
        )
        return state, beliefs

    def step(self, key: jax.Array, state, beliefs, actions: Actions):
        """One env step + per-side belief update.

        Returns (next_state, next_beliefs, step_output).
        """
        k_env, k_g_obs, k_b_obs, k_g_upd, k_b_upd = jax.random.split(key, 5)
        step_out = self.env.step(k_env, state, actions)
        guard_obs = self.env.guard_observation_fn(
            step_out.state, Side.GUARD, self.env.config, k_g_obs, step_out.state.t
        )
        bandit_obs = self.env.bandit_observation_fn(
            step_out.state, Side.BANDIT, self.env.config, k_b_obs, step_out.state.t
        )
        next_beliefs = BySide(
            guard=self.guard_belief_updater(
                beliefs.guard, guard_obs, actions.sides.guard, Side.GUARD, k_g_upd
            ),
            bandit=self.bandit_belief_updater(
                beliefs.bandit, bandit_obs, actions.sides.bandit, Side.BANDIT, k_b_upd
            ),
        )
        return step_out.state, next_beliefs, step_out
