"""CompositeActionPolicy — sum of learned offset and scripted base actions."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.policies import ZeroControl
from orbital_game.policies.controlled import CompositeActionPolicy


@dataclass(frozen=True)
class _ConstantOffset:
    """Test fixture: emits a fixed +x offset action."""

    n_vehicles: int = 1
    action_dim: int = 3

    def __call__(self, policy_state, obs, key, t):
        return jnp.broadcast_to(
            jnp.array([0.05, 0.0, 0.0]), (self.n_vehicles, self.action_dim)
        ), policy_state


def test_composite_sums_base_and_offset_actions():
    offset = _ConstantOffset()
    base = ZeroControl(n_vehicles=1, action_dim=3)
    p = CompositeActionPolicy(base=base, offset=offset, n_vehicles=1, action_dim=3)
    obs = jnp.zeros(12)
    action, _ = p(None, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    # ZeroControl gives 0; _ConstantOffset gives [0.05, 0, 0]; sum = [0.05, 0, 0].
    assert jnp.array_equal(action[0], jnp.array([0.05, 0.0, 0.0]))
