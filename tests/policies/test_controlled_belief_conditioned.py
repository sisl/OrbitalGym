"""BeliefConditionedPolicy — prepends a belief vector to the obs."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from orbital_game.policies.controlled import BeliefConditionedPolicy


@dataclass(frozen=True)
class _ObsRecorder:
    """Test fixture: returns the obs back as the action's first row, padded."""

    n_vehicles: int = 1
    action_dim: int = 3

    def __call__(self, policy_state, obs, key, t):
        # Pack the first 3 obs floats into an action so the test can read them back.
        action = jnp.broadcast_to(obs[:3], (self.n_vehicles, self.action_dim))
        return action, policy_state


def test_belief_is_prepended_to_obs_for_base_policy():
    base = _ObsRecorder()
    p = BeliefConditionedPolicy(base=base, n_vehicles=1, action_dim=3)
    obs = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    belief = jnp.array([1.0, 2.0, 3.0])
    state = (belief, None)  # (belief_mean, base_state)
    action, _ = p(state, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    # Base policy reads obs[:3]; with prepend, the augmented obs[:3] should be
    # the belief vector.
    assert jnp.array_equal(action[0], belief)
