"""RecedingHorizonRollout chooses an action that beats random on average."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from examples.policies.receding_horizon import RecedingHorizonRollout
from orbitalgym import OrbitalGymEnv, Side, make_pursuit_evasion
from orbitalgym.adapters.pomdp import POMDPAdapter


def test_receding_horizon_beats_random_on_pursuit_evasion():
    cfg = make_pursuit_evasion(seed=0)
    adapter = POMDPAdapter(OrbitalGymEnv(cfg))
    planner = RecedingHorizonRollout(
        adapter=adapter,
        controlled_side=Side.BANDIT,
        n_candidates=32,
        horizon=3,
        dv_scale=0.05,
    )

    key = jax.random.PRNGKey(0)
    s0 = adapter.initialstate(key)

    a_planner = planner.act(s0, jax.random.PRNGKey(1))

    # Reward of best-found action vs reward of a random action averaged
    # over a few seeds. Planner should produce a higher expected
    # one-step reward.
    n_g = cfg.n_guards
    n_b = cfg.n_bandits
    d = adapter.action_dim_per_side

    def random_action(seed):
        k = jax.random.PRNGKey(seed)
        return jax.random.normal(k, ((n_g + n_b) * d,)) * 0.05

    rewards_random = []
    rewards_planner = []
    for seed in range(5):
        k_rand = jax.random.PRNGKey(100 + seed)
        s_next_random = adapter.transition(s0, random_action(seed), k_rand)
        s_next_planner = adapter.transition(s0, a_planner, k_rand)
        rewards_random.append(
            float(adapter.reward(s0, random_action(seed), s_next_random, Side.BANDIT))
        )
        rewards_planner.append(float(adapter.reward(s0, a_planner, s_next_planner, Side.BANDIT)))

    assert jnp.mean(jnp.array(rewards_planner)) >= jnp.mean(jnp.array(rewards_random)), (
        f"planner mean {float(jnp.mean(jnp.array(rewards_planner))):.4f} "
        f"< random mean {float(jnp.mean(jnp.array(rewards_random))):.4f}"
    )
