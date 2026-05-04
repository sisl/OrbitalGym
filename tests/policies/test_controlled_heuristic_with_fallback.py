"""HeuristicWithFallbackPolicy — routes between a primary and a fallback
based on a confidence scalar threaded through `policy_state`."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.policies import ZeroControl
from orbital_game.policies.controlled import HeuristicWithFallbackPolicy


@dataclass(frozen=True)
class _ConstantConfidencePrimary:
    """Test fixture: emits +1 thrust on +x. policy_state is a confidence float."""

    n_vehicles: int = 1
    action_dim: int = 3

    def __call__(self, policy_state, obs, key, t):
        return jnp.broadcast_to(
            jnp.array([1.0, 0.0, 0.0]), (self.n_vehicles, self.action_dim)
        ), policy_state


def test_routes_to_primary_when_confidence_above_threshold():
    primary = _ConstantConfidencePrimary()
    fallback = ZeroControl(n_vehicles=1, action_dim=3)
    p = HeuristicWithFallbackPolicy(
        primary=primary, fallback=fallback, threshold=0.5, n_vehicles=1, action_dim=3
    )
    obs = jnp.zeros(12)
    confidence = jnp.asarray(0.9)
    action, _ = p(confidence, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert float(action[0, 0]) == 1.0


def test_routes_to_fallback_when_confidence_below_threshold():
    primary = _ConstantConfidencePrimary()
    fallback = ZeroControl(n_vehicles=1, action_dim=3)
    p = HeuristicWithFallbackPolicy(
        primary=primary, fallback=fallback, threshold=0.5, n_vehicles=1, action_dim=3
    )
    obs = jnp.zeros(12)
    confidence = jnp.asarray(0.1)
    action, _ = p(confidence, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    assert float(action[0, 0]) == 0.0
