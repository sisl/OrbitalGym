"""Tests for rollout.py — jax.lax.scan-based rollout + vmap over batched seeds."""

import jax
import jax.numpy as jnp

from orbital_game.config import ScenarioConfig, VehicleParamsSpec
from orbital_game.env.environment import OrbitalGameEnv
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.registry import StateComponentKey
from orbital_game.rollout import Trajectory, episode_mask, rollout
from orbital_game.sampling.mass import ConstantMass
from orbital_game.sampling.side import RelativeEllipse
from orbital_game.sampling.spec import ICSpec


def _make_cfg(max_horizon_s: float = 2000.0) -> ScenarioConfig:
    return ScenarioConfig(
        n_guards=1,
        n_bandits=1,
        epoch_mjd_utc=60067.0,
        reference_orbit=ReferenceOrbitState(
            position_eci=jnp.array([7000e3, 0.0, 0.0]),
            velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
        ),
        guard_components=(StateComponentKey.RTN, StateComponentKey.MASS),
        bandit_components=(StateComponentKey.RTN,),
        guard_params=VehicleParamsSpec(100.0, 220.0, 5.0),
        bandit_params=VehicleParamsSpec(50.0, 200.0, 2.0),
        ic_sampler=ICSpec(
            guard_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
                # phase_rad=None -> uniform random per vehicle, so different
                # seeds produce different ICs (needed for the vmap test).
                mass_sampler=ConstantMass(propellant_mass_kg=10.0),
            ),
            bandit_sampler=RelativeEllipse(
                radial_ellipse_m=1000.0,
            ),
        ),
        dt=10.0,
        max_horizon_s=max_horizon_s,
        seed=0,
    )


def _make_env(max_horizon_s: float = 2000.0) -> OrbitalGameEnv:
    return OrbitalGameEnv(_make_cfg(max_horizon_s=max_horizon_s))


def _zero_policy(policy_state, obs, key, t):
    del obs, key, t
    return jnp.zeros((1, 3)), policy_state


def _null_init_policy_state(config, env_state, key):
    del config, env_state, key
    return None


def test_rollout_returns_trajectory_with_leading_T_axis():  # noqa: N802
    env = _make_env()
    traj = rollout(env, _zero_policy, _null_init_policy_state, jax.random.PRNGKey(0), n_steps=20)
    assert traj.reward.shape == (20,)
    assert traj.action.shape == (20, 1, 3)
    assert traj.done.shape == (20,)


def test_rollout_is_deterministic_under_same_key():
    env = _make_env()
    key = jax.random.PRNGKey(7)
    t1 = rollout(env, _zero_policy, _null_init_policy_state, key, n_steps=10)
    t2 = rollout(env, _zero_policy, _null_init_policy_state, key, n_steps=10)
    assert jnp.allclose(t1.reward, t2.reward)


def test_vmap_rollout_over_seeds_produces_batched_trajectories():
    env = _make_env()
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    batched = jax.vmap(lambda k: rollout(env, _zero_policy, _null_init_policy_state, k, n_steps=10))
    t = batched(keys)
    assert t.reward.shape == (4, 10)
    # Different seeds → different rewards (sanity check that the vmap axis is real).
    assert not jnp.allclose(t.reward[0], t.reward[1])


def test_rollout_freezes_state_and_zeros_reward_after_termination():
    """max_horizon_s=30 with dt=10 → max_steps=3. Termination should fire at the
    step where env_state.step reaches 3 (scan_index=2: env.step #3 advances
    state.step 2→3, MaxStepsOrBreach sees 3>=3 → done). From scan_index=3
    onward, the rollout freezes: logged env_state.step stays at 3, reward 0,
    done latched True."""
    env = _make_env(max_horizon_s=30.0)  # max_steps = 3
    traj = rollout(env, _zero_policy, _null_init_policy_state, jax.random.PRNGKey(0), n_steps=10)

    # Logged env_state.step at each scan index (pre-step input state):
    # [0, 1, 2, 3, 3, 3, 3, 3, 3, 3]
    expected_step = jnp.array([0, 1, 2, 3, 3, 3, 3, 3, 3, 3])
    assert jnp.array_equal(traj.env_state.step, expected_step)

    # Done first fires at scan_index=2 (where env.step #3 produces state.step=3).
    assert not traj.done[0]
    assert not traj.done[1]
    assert traj.done[2]
    # Done latched True from scan_index=2 onward.
    assert jnp.all(traj.done[2:])

    # Reward at the terminating step (scan_index=2) is the actual value; zeros
    # from scan_index=3 onward.
    assert jnp.all(traj.reward[3:] == 0.0)


def test_episode_mask_marks_steps_up_to_and_including_first_termination():
    """mask[t] = True iff no done has occurred at any step < t."""
    done = jnp.array([False, False, True, True, True, True, True, True, True, True])
    traj = Trajectory(
        env_state=None,
        action=jnp.zeros((10, 1, 3)),
        reward=jnp.zeros((10,)),
        done=done,
        obs=jnp.zeros((10, 1)),
        policy_state=None,
    )
    expected = jnp.array([True, True, True, False, False, False, False, False, False, False])
    assert jnp.array_equal(episode_mask(traj), expected)


def test_episode_mask_is_all_true_when_no_termination():
    done = jnp.zeros((10,), dtype=bool)
    traj = Trajectory(
        env_state=None,
        action=jnp.zeros((10, 1, 3)),
        reward=jnp.zeros((10,)),
        done=done,
        obs=jnp.zeros((10, 1)),
        policy_state=None,
    )
    assert jnp.all(episode_mask(traj))


def test_episode_mask_works_under_vmap():
    """episode_mask must handle batched (B, T) shape, not just (T,)."""
    done = jnp.array(
        [
            [False, False, True, True, True],
            [False, True, True, True, True],
            [False, False, False, False, False],
        ]
    )
    traj = Trajectory(
        env_state=None,
        action=jnp.zeros((3, 5, 1, 3)),
        reward=jnp.zeros((3, 5)),
        done=done,
        obs=jnp.zeros((3, 5, 1)),
        policy_state=None,
    )
    expected = jnp.array(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
            [True, True, True, True, True],
        ]
    )
    assert jnp.array_equal(episode_mask(traj), expected)
