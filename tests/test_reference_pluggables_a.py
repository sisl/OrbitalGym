"""Tests for reference intruder policy, observation, reward."""

import jax
import jax.numpy as jnp

from orbital_game.observations.reference import FullObservation
from orbital_game.policies.intruder import ZeroControlIntruder
from orbital_game.rewards.reference import DistanceToHVA


def test_zero_control_intruder_returns_zeros():
    pol = ZeroControlIntruder(n_intruders=3, action_dim=3)
    out = pol(obs=None, key=jax.random.PRNGKey(0), t=jnp.array(0.0))
    assert out.shape == (3, 3)
    assert jnp.allclose(out, 0.0)


def test_full_observation_returns_flat_state_vector():
    class _Layout:
        def flatten(self, d, i):
            return jnp.concatenate([d, i])

    class _Env:
        defenders = jnp.arange(6.0)
        intruders = jnp.arange(6.0) + 100

    obs_fn = FullObservation(layout=_Layout())
    out = obs_fn(
        env_state=_Env(),
        params=None,
        key=jax.random.PRNGKey(0),
        t=jnp.array(0.0),
        for_side="defender",
    )
    assert jnp.allclose(out, jnp.concatenate([jnp.arange(6.0), jnp.arange(6.0) + 100]))


def test_distance_to_hva_reward_is_negative_distance():
    """Reference reward: sum of negative Euclidean distances from each defender to HVA."""

    class _Def:
        rtn = jnp.array([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0]])

    class _Env:
        defenders = _Def()

    rw = DistanceToHVA()
    r = rw(prev_state=None, action=None, next_state=_Env(), params=None, t=jnp.array(0.0))
    assert jnp.isclose(r, -(100.0 + 200.0))
