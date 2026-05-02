"""T4 — Plan with short-horizon search. Source-of-truth for snippets in
docs/tutorials/t4-short-horizon-search.md.
"""

from __future__ import annotations


def test_t4_short_horizon_search_walkthrough():
    # --8<-- [start:imports]
    from dataclasses import dataclass

    import jax
    import jax.numpy as jnp

    from orbital_game import (
        OrbitalGameEnv,
        POMDPAdapter,
        Side,
        make_pursuit_evasion,
    )
    # --8<-- [end:imports]

    # --8<-- [start:adapter-setup]
    cfg = make_pursuit_evasion(seed=0)
    env = OrbitalGameEnv(cfg)
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

        ``POMDPAdapter`` is *stateful*: ``transition`` mutates
        ``adapter._last_state`` to thread the env's ``t``/``step``/
        ``reference_orbit`` through the flat-vector API. That means
        candidate rollouts must run in a plain Python loop — wrapping
        ``transition`` in ``jax.vmap`` or ``jax.lax.scan`` would leak
        tracers into ``_last_state`` and break the next call. Snapshot
        ``_last_state`` once before the search, restore it between
        candidates so each rollout starts from the same root, and
        restore it again at exit so callers see no side effect.
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

            # Snapshot the adapter's mutable state. Each candidate
            # rollout will reset to this point before stepping forward.
            saved_last_state = self.adapter._last_state

            seq_keys = jax.random.split(key, self.n_candidates)
            best_score = -jnp.inf
            best_first_action = jnp.zeros(action_dim)

            for cand_idx in range(self.n_candidates):
                keys = jax.random.split(seq_keys[cand_idx], self.horizon + 1)
                actions = jax.random.normal(keys[0], (self.horizon, action_dim)) * self.dv_scale

                # Reset the adapter to the planning root before this
                # candidate's rollout.
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

            # Restore so callers see no observable mutation.
            self.adapter._last_state = saved_last_state
            return best_first_action

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
