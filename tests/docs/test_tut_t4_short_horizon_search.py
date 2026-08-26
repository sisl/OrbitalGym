"""T4 — Plan with short-horizon search. Source-of-truth for snippets in
docs/tutorials/t4-short-horizon-search.md.
"""

from __future__ import annotations


def test_t4_short_horizon_search_walkthrough():
    # --8<-- [start:imports]
    from dataclasses import dataclass

    import jax
    import jax.numpy as jnp

    from orbitalgym import (
        OrbitalGymEnv,
        POMDPAdapter,
        Side,
        make_pursuit_evasion,
    )
    # --8<-- [end:imports]

    # --8<-- [start:adapter-setup]
    cfg = make_pursuit_evasion(seed=0)
    env = OrbitalGymEnv(cfg)
    adapter = POMDPAdapter(env)

    states_dim = adapter.states_dim
    action_dim_per_side = adapter.action_dim_per_side
    n_g = cfg.n_guards
    n_b = cfg.n_bandits
    full_action_dim = (n_g + n_b) * action_dim_per_side
    # --8<-- [end:adapter-setup]

    # --8<-- [start:initial-state]
    s0 = adapter.initialstate(jax.random.PRNGKey(0))
    # --8<-- [end:initial-state]

    # --8<-- [start:planner-class]
    @dataclass(frozen=True)
    class RecedingHorizonRollout:
        """Random-shooting planner over a flat-vector POMDPAdapter.

        Sample ``n_candidates`` action sequences of length ``horizon``,
        simulate each forward through ``adapter.transition``, score by
        cumulative reward to ``controlled_side``, and return the first
        action of the best-scoring sequence.

        ``POMDPAdapter`` is pure, so the entire search compiles to a
        single ``jax.vmap(jax.lax.scan)`` call: candidates run in
        parallel on whatever device JAX has, and each candidate's
        horizon rolls forward inside a scanned step.
        """

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

    # --8<-- [end:planner-class]

    # --8<-- [start:plan-one-step]
    planner = RecedingHorizonRollout(
        adapter=adapter,
        controlled_side=Side.BANDIT,
        n_candidates=8,
        horizon=3,
    )
    key_plan, key_step = jax.random.split(jax.random.PRNGKey(1))
    a_planner = planner.act(s0, key_plan)
    s1 = adapter.transition(s0, a_planner, key_step)
    r_bandit = adapter.reward(s0, a_planner, s1, Side.BANDIT)
    # --8<-- [end:plan-one-step]

    assert s0.shape == (states_dim,)
    assert a_planner.shape == (full_action_dim,)
    assert s1.shape == (states_dim,)
    assert jnp.isfinite(r_bandit)
