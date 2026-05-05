"""BeliefConditionedPolicy — prepends a belief vector to the obs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbital_game.policies.controlled import BeliefConditionedPolicy
from tests.policies._helpers import make_impulsive_maneuver_command_cls


@dataclass(frozen=True)
class _ObsRecorder:
    """Test fixture: returns the obs back as the action's first row, padded."""

    n_vehicles: int = 1
    command_cls: Any = None

    def __call__(self, policy_state, obs, key, t):
        # Pack the first 3 obs floats into the command's dv field so the test
        # can read them back.
        dv = jnp.broadcast_to(obs[:3], (self.n_vehicles, 3))
        cmd = self.command_cls.zeros(self.n_vehicles).replace(dv=dv)
        return cmd, policy_state


def test_belief_is_prepended_to_obs_for_base_policy():
    cmd_cls = make_impulsive_maneuver_command_cls(1)
    base = _ObsRecorder(n_vehicles=1, command_cls=cmd_cls)
    p = BeliefConditionedPolicy(base=base, n_vehicles=1, command_cls=cmd_cls)
    obs = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    belief = jnp.array([1.0, 2.0, 3.0])
    state = (belief, None)  # (belief_mean, base_state)
    cmd, _ = p(state, obs, jax.random.PRNGKey(0), jnp.asarray(0))
    # Base policy reads obs[:3]; with prepend, the augmented obs[:3] should be
    # the belief vector.
    assert jnp.array_equal(cmd.dv[0], belief)
