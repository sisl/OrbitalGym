"""Initial-condition bank: sample, save, load, index."""

import jax
import jax.numpy as jnp

from orbitalgym import OrbitalGymEnv, make_lady_bandit_guard
from orbitalgym.eval.bank import bank_episode, load_bank, sample_bank, save_bank


def test_sample_bank_has_leading_axis_and_distinct_episodes():
    env = OrbitalGymEnv(make_lady_bandit_guard())
    states = sample_bank(env, n_episodes=4, seed=2027)
    assert states.guards.rtn.shape == (4, 1, 6)
    assert states.ic_valid.shape == (4,)
    assert not jnp.allclose(states.guards.rtn[0], states.guards.rtn[1])


def test_sample_bank_is_deterministic_in_seed():
    env = OrbitalGymEnv(make_lady_bandit_guard())
    a = sample_bank(env, n_episodes=3, seed=1)
    b = sample_bank(env, n_episodes=3, seed=1)
    assert jnp.array_equal(a.bandits.rtn, b.bandits.rtn)


def test_bank_round_trips_through_hdf5(tmp_path):
    cfg = make_lady_bandit_guard()
    env = OrbitalGymEnv(cfg)
    states = sample_bank(env, n_episodes=4, seed=5)
    path = tmp_path / "bank.h5"
    save_bank(path, cfg, states, seed=5)
    cfg2, states2 = load_bank(path)
    assert cfg2.n_guards == cfg.n_guards
    assert jnp.array_equal(states2.guards.rtn, states.guards.rtn)
    assert jnp.array_equal(
        states2.reference_orbit.position_eci, states.reference_orbit.position_eci
    )
    assert jnp.array_equal(states2.ic_valid, states.ic_valid)


def test_bank_episode_indexes_one_state():
    env = OrbitalGymEnv(make_lady_bandit_guard())
    states = sample_bank(env, n_episodes=4, seed=5)
    one = bank_episode(states, 2)
    assert one.guards.rtn.shape == (1, 6)
    assert jnp.array_equal(one.guards.rtn, states.guards.rtn[2])
    restored, _ = env.reset_from_state(one, jax.random.PRNGKey(0))
    assert float(restored.t) == 0.0
