"""reset_from_state restarts an episode from a stored EnvState."""

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.env.types import BySide
from orbitalgym.policies.zero import ZeroControl
from orbitalgym.rollout import rollout


def test_reset_from_state_reproduces_positions_and_zeroes_clock():
    env = OrbitalGymEnv(make_lady_bandit_guard())
    state, _ = env.reset(jax.random.PRNGKey(3))
    advanced = state.replace(t=jnp.asarray(50.0), step=jnp.asarray(5))
    restored, outputs = env.reset_from_state(advanced, jax.random.PRNGKey(0))
    assert float(restored.t) == 0.0
    assert int(restored.step) == 0
    assert jnp.allclose(restored.guards.rtn, state.guards.rtn)
    assert jnp.allclose(restored.bandits.rtn, state.bandits.rtn)
    expected_obs = env.reset(jax.random.PRNGKey(0))[1].guard.obs[0].obs.shape
    assert outputs.guard.obs[0].obs.shape == expected_obs


def test_rollout_from_initial_state_starts_there():
    env = OrbitalGymEnv(make_lady_bandit_guard())
    state, _ = env.reset(jax.random.PRNGKey(7))
    policies = BySide(
        guard=ZeroControl(n_vehicles=1, command_cls=env.guard_command_cls),
        bandit=ZeroControl(n_vehicles=1, command_cls=env.bandit_command_cls),
    )
    init = BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None)
    traj = rollout(env, policies, init, jax.random.PRNGKey(99), n_steps=3, initial_state=state)
    assert jnp.allclose(traj.env_state.guards.rtn[0], state.guards.rtn)
    assert jnp.allclose(traj.env_state.bandits.rtn[0], state.bandits.rtn)
