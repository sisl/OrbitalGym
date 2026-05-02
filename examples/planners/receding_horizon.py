"""RecedingHorizonRollout — short-horizon random-shooting planner.

For a given controlled side, sample K candidate action sequences of
horizon H, simulate each via POMDPAdapter.transition, score by
cumulative reward to the controlled side, and return the first action
of the highest-scoring sequence.

Not MCTS — no tree, no expansion-by-uncertainty. The teaching value is
the API patterns: how to call transition / reward, how to thread keys,
how the action vector is laid out.

Note: POMDPAdapter is stateful (it stashes ``_last_state`` between calls
to thread the env's t/step/reference_orbit through the flat-vector API),
so this planner uses plain Python loops rather than ``jax.vmap`` /
``jax.lax.scan``. JAX transforms over a side-effecting adapter would
leak tracers. The teaching point: the POMDPPlanners-shape interface is
imperative, not pure.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game import Side
from orbital_game.adapters.pomdp import POMDPAdapter


@dataclass(frozen=True)
class RecedingHorizonRollout:
    adapter: POMDPAdapter
    controlled_side: Side
    n_candidates: int = 32
    horizon: int = 3
    dv_scale: float = 0.05

    def act(self, s_flat: jax.Array, key: jax.Array) -> jax.Array:
        n_g = self.adapter.env.config.n_guards
        n_b = self.adapter.env.config.n_bandits
        d = self.adapter.action_dim_per_side
        action_dim = (n_g + n_b) * d

        # Snapshot the adapter's internal state so each candidate rollout
        # starts from the same initial point. ``transition`` mutates
        # ``_last_state``; we restore it between candidates and at exit.
        saved_last_state = self.adapter._last_state

        seq_keys = jax.random.split(key, self.n_candidates)
        best_score = -jnp.inf
        best_first_action = jnp.zeros(action_dim)

        for cand_idx in range(self.n_candidates):
            keys = jax.random.split(seq_keys[cand_idx], self.horizon + 1)
            actions = jax.random.normal(keys[0], (self.horizon, action_dim)) * self.dv_scale

            # Reset adapter to the planning root before each rollout.
            self.adapter._last_state = saved_last_state

            s = s_flat
            total_reward = jnp.asarray(0.0)
            for h in range(self.horizon):
                a = actions[h]
                s_next = self.adapter.transition(s, a, keys[h + 1])
                r = self.adapter.reward(s, a, s_next, self.controlled_side)
                total_reward = total_reward + r
                s = s_next

            if float(total_reward) > float(best_score):
                best_score = total_reward
                best_first_action = actions[0]

        # Restore adapter's state so callers see no observable mutation.
        self.adapter._last_state = saved_last_state
        return best_first_action
