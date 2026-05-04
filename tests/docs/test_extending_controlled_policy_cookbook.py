"""Controlled-side policy cookbook — source-of-truth for snippets in
docs/extending/controlled-policy-cookbook.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def test_controlled_cookbook_walkthrough():
    # --8<-- [start:imports]

    from orbital_game.policies import ZeroControl
    from orbital_game.policies.controlled import (
        BeliefConditionedPolicy,
        CompositeActionPolicy,
        HeuristicWithFallbackPolicy,
    )
    # --8<-- [end:imports]

    # --8<-- [start:scripted-with-fallback-pattern]
    @dataclass(frozen=True)
    class _Primary:
        n_vehicles: int = 1
        action_dim: int = 3

        def __call__(self, ps, obs, key, t):
            return jnp.broadcast_to(
                jnp.array([0.05, 0.0, 0.0]), (self.n_vehicles, self.action_dim)
            ), ps

    primary = _Primary()
    fallback = ZeroControl(n_vehicles=1, action_dim=3)
    policy = HeuristicWithFallbackPolicy(
        primary=primary,
        fallback=fallback,
        threshold=0.5,
        n_vehicles=1,
        action_dim=3,
    )
    # Pass confidence through policy_state. High confidence → primary.
    confident = jnp.asarray(0.9)
    action_high, _ = policy(confident, jnp.zeros(12), jax.random.PRNGKey(0), jnp.asarray(0))
    # Low confidence → fallback (ZeroControl).
    unconfident = jnp.asarray(0.1)
    action_low, _ = policy(unconfident, jnp.zeros(12), jax.random.PRNGKey(0), jnp.asarray(0))
    # --8<-- [end:scripted-with-fallback-pattern]

    assert float(action_high[0, 0]) == 0.05
    assert float(action_low[0, 0]) == 0.0

    # --8<-- [start:composite-action-pattern]
    @dataclass(frozen=True)
    class _LearnedOffset:
        n_vehicles: int = 1
        action_dim: int = 3

        def __call__(self, ps, obs, key, t):
            # In real use: a NN forward pass.
            return jnp.broadcast_to(
                jnp.array([0.01, 0.0, 0.0]), (self.n_vehicles, self.action_dim)
            ), ps

    base = ZeroControl(n_vehicles=1, action_dim=3)
    composite = CompositeActionPolicy(
        base=base, offset=_LearnedOffset(), n_vehicles=1, action_dim=3
    )
    action, _ = composite(None, jnp.zeros(12), jax.random.PRNGKey(0), jnp.asarray(0))
    # --8<-- [end:composite-action-pattern]

    assert float(action[0, 0]) == 0.01

    # --8<-- [start:belief-conditioned-pattern]
    @dataclass(frozen=True)
    class _BeliefAware:
        """Sees augmented obs (belief prepended) and uses the belief mean."""

        n_vehicles: int = 1
        action_dim: int = 3

        def __call__(self, ps, obs, key, t):
            # First 3 obs floats are the belief mean (prepended by wrapper).
            return jnp.broadcast_to(obs[:3] * 0.001, (self.n_vehicles, self.action_dim)), ps

    base_aware = _BeliefAware()
    wrapped = BeliefConditionedPolicy(base=base_aware, n_vehicles=1, action_dim=3)
    obs_raw = jnp.zeros(6)
    belief_mean = jnp.array([10.0, 20.0, 30.0])
    state = (belief_mean, None)  # (belief_mean, base_state)
    bc_action, _ = wrapped(state, obs_raw, jax.random.PRNGKey(0), jnp.asarray(0))
    # --8<-- [end:belief-conditioned-pattern]

    assert jnp.allclose(bc_action[0], jnp.array([0.01, 0.02, 0.03]))
