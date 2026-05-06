"""RecedingHorizonRollout — short-horizon random-shooting planner.

For a given controlled side, sample K candidate action sequences of
horizon H, simulate each via ``POMDPAdapter.transition``, score by
cumulative reward to the controlled side, and return the first action
of the highest-scoring sequence.

Pure JAX: ``jax.vmap`` over candidates, ``jax.lax.scan`` over horizon —
the entire search runs as one compiled call. The teaching value is
the API patterns: how to call ``transition`` / ``reward``, how to
thread keys, how the action vector is laid out, and how a pure
adapter composes with JAX transforms.
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

        def score_sequence(seq_key):
            keys = jax.random.split(seq_key, self.horizon + 1)
            actions = jax.random.normal(keys[0], (self.horizon, action_dim)) * self.dv_scale

            def step(carry_state, idx):
                a = actions[idx]
                s_next = self.adapter.transition(carry_state, a, keys[idx + 1])
                r = self.adapter.reward(carry_state, a, s_next, self.controlled_side)
                return s_next, r

            _, rewards = jax.lax.scan(step, s_flat, jnp.arange(self.horizon))
            return rewards.sum(), actions[0]

        seq_keys = jax.random.split(key, self.n_candidates)
        scores, first_actions = jax.vmap(score_sequence)(seq_keys)
        best = jnp.argmax(scores)
        return first_actions[best]
