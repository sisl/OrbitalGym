"""jax.lax.scan-based rollout driving an OrbitalGameEnv for n_steps.

Design notes:
  - The rollout uses the observation returned by env.reset / env.step directly;
    no extra call to env.defender_observation_fn is needed. The scan carry
    threads (env_state, policy_state, obs, terminated) so the policy at step t
    sees the observation from step t-1 (or env.reset for step 0).
  - **Freeze-on-done.** Once `done=True` fires at step k, subsequent steps
    freeze: env_state / obs / policy_state hold at step-k's values, reward is
    zero, done stays True. The scan still runs for `n_steps` (JAX requires a
    static length) but post-termination outputs are semantically clean.
  - Deterministic under a single master PRNGKey. Seed-parallel rollouts come
    from `jax.vmap(rollout, in_axes=(None, None, None, 0, None))` over a batch
    of keys. Different batch elements can terminate at different times — the
    freeze-on-done logic handles heterogeneous termination uniformly.
  - `episode_mask(trajectory)` returns a boolean mask over valid (pre+at
    termination) steps — see §A in the docstring below.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class Trajectory:
    """Time-stacked record of a rollout. Each field has leading axis (T,)
    (or (B, T, ...) after vmap over seeds).

    Once an episode terminates (first done=True at some step k):
      - env_state / obs / policy_state at step > k hold at step-k values (frozen)
      - reward at step > k is zero (the terminating step k still gets its reward)
      - done stays True from step k onwards (latched)
    Use `episode_mask(traj)` to get a boolean mask over valid steps.
    """

    env_state: Any  # EnvState pytree, leading T axis per leaf
    action: jax.Array  # (T, N_defenders, action_dim)
    reward: jax.Array  # (T,)
    done: jax.Array  # (T,) boolean — latched True after first termination
    obs: jax.Array  # (T, obs_dim) — defender obs the policy saw at step t
    policy_state: Any  # policy_state pytree at each step, leading T axis


def rollout(
    env,
    defender_policy: Callable,
    init_policy_state_fn: Callable,
    key: jax.Array,
    n_steps: int,
) -> Trajectory:
    """Run a deterministic seed-driven rollout for n_steps, with freeze-on-done.

    Order:
      1. Split key into (k_reset, k_init_ps, k_scan).
      2. env.reset(k_reset) -> (env_state, initial_obs).
      3. init_policy_state_fn(env.config, env_state, k_init_ps) -> policy_state.
      4. scan n_steps: policy(ps, obs) -> action; env.step -> (next_es, next_obs, r, d).
         If already terminated coming in, freeze state/obs/ps and zero reward.

    Args:
      env: OrbitalGameEnv
      defender_policy: (policy_state, obs, key, t) -> (action, next_policy_state)
      init_policy_state_fn: (config, env_state, key) -> policy_state
      key: master PRNGKey for the rollout
      n_steps: scan length (fixed; jax.lax.scan requires a static loop count)
    """
    k_reset, k_init_ps, k_scan = jax.random.split(key, 3)
    env_state, initial_obs = env.reset(k_reset)
    policy_state = init_policy_state_fn(env.config, env_state, k_init_ps)
    initial_terminated = jnp.asarray(False)

    def _step(carry, step_key):
        es, ps, obs, terminated = carry
        k_act, k_env = jax.random.split(step_key, 2)
        action, next_ps = defender_policy(ps, obs, k_act, es.t)
        next_es, next_obs, reward, done, _ = env.step(k_env, es, action)

        next_terminated = terminated | done

        # If ALREADY terminated coming into this step, hold state/obs/ps at
        # their current (frozen) values. Otherwise advance normally. The
        # terminating step itself (the first step with done=True) logs real
        # values and its next_es becomes the frozen reference for later steps.
        advance_es = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), es, next_es
        )
        advance_obs = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), obs, next_obs
        )
        advance_ps = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), ps, next_ps
        )

        # Zero reward AFTER termination. The terminating step itself accrues
        # its natural reward (terminated was False coming in).
        out_reward = jnp.where(terminated, jnp.zeros_like(reward), reward)

        logged = {
            "env_state": es,
            "action": action,
            "reward": out_reward,
            "done": next_terminated,
            "obs": obs,
            "policy_state": ps,
        }
        return (advance_es, advance_ps, advance_obs, next_terminated), logged

    step_keys = jax.random.split(k_scan, n_steps)
    _, stacked = jax.lax.scan(
        _step,
        (env_state, policy_state, initial_obs, initial_terminated),
        step_keys,
    )

    return Trajectory(
        env_state=stacked["env_state"],
        action=stacked["action"],
        reward=stacked["reward"],
        done=stacked["done"],
        obs=stacked["obs"],
        policy_state=stacked["policy_state"],
    )


def episode_mask(traj: Trajectory) -> jax.Array:
    """Return a boolean mask marking steps up to and including the terminating
    step (False for post-terminal steps).

    Shape matches `traj.done` (either `(T,)` or `(B, T)` under vmap). Semantics:

      mask[t] = True  iff no `done=True` occurred at any step < t
                      (equivalently: step t is before OR AT the first termination)

    Example — traj.done = [F, F, T, T, T]:
      mask   = [T, T, T, F, F]   (step 2 is the terminating step — still valid)

    Useful cases:
      - Episode length: `jnp.sum(mask)` (along the time axis).
      - Masking action/obs/env_state for analysis: freeze-on-done zeros reward
        and freezes state/obs, but `action` is logged verbatim — use the mask
        when you want action statistics over valid steps only.
      - Working with trajectories from rollouts that do NOT use freeze-on-done.
    """
    done = traj.done
    # Shift done right by one step along the time axis (prepend False).
    shifted = jnp.concatenate(
        [jnp.zeros_like(done[..., :1]), done[..., :-1]],
        axis=-1,
    )
    cumulative_any_before = jnp.cumsum(shifted, axis=-1) > 0
    return ~cumulative_any_before
