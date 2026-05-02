"""jax.lax.scan-based rollout for the symmetric core.

Both sides have policies; the scan threads (env_state, BySide[policy_state],
BySide[obs], terminated_flag). Each step:

  1. For each side, policy maps (ps_s, obs_s, key, t) → (action_s, ps_s').
  2. Bundle actions as Actions; call env.step → StepOutput.
  3. Apply freeze-on-done logic per side (state/obs/ps held; reward zeroed).

For single-agent runs, callers typically use rollout_single_agent which
wraps SingleAgentView.

Key-split note: the top-level split uses split(key, 3) → (k_reset, k_init,
k_scan) to preserve byte-identical numerical output with the Phase-0 baseline
fixture. k_init is further split into (k_init_g, k_init_b). Since all current
observation functions and the dynamics step are deterministic (keys discarded),
only k_reset affects numerical output via IC sampling.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from orbital_game.env.types import (
    Actions,
    BySide,
    Side,
    SideTrajectory,
    Trajectory,
)


def rollout(
    env,
    policies: BySide,                # BySide[Policy] — one policy per side
    init_policy_state_fns: BySide,   # BySide[Callable(config, env_state, key) -> ps]
    key: jax.Array,
    n_steps: int,
) -> Trajectory:
    """Symmetric rollout. Both sides driven by their own policies.

    Returns a Trajectory with sides.guard / sides.bandit each carrying
    (T, N_side, ...) leading-axis arrays, plus episode_done (T,) latched.

    Key split: k_reset, k_init, k_scan = split(key, 3) to preserve byte
    identity with the Phase-0 baseline (only k_reset → k_ic → IC sampling
    matters; all other keys are discarded by deterministic callables).
    """
    k_reset, k_init, k_scan = jax.random.split(key, 3)
    k_init_g, k_init_b = jax.random.split(k_init, 2)
    env_state, initial_outputs = env.reset(k_reset)
    ps_g = init_policy_state_fns.guard(env.config, env_state, k_init_g)
    ps_b = init_policy_state_fns.bandit(env.config, env_state, k_init_b)
    initial_terminated = jnp.asarray(False)

    def _step(carry, step_key):
        es, ps_g, ps_b, obs_g, obs_b, terminated = carry
        k_act_g, k_act_b, k_env = jax.random.split(step_key, 3)
        action_g, next_ps_g = policies.guard(ps_g, obs_g, k_act_g, es.t)
        action_b, next_ps_b = policies.bandit(ps_b, obs_b, k_act_b, es.t)
        actions = Actions(sides=BySide(guard=action_g, bandit=action_b))
        step_out = env.step(k_env, es, actions)
        next_es = step_out.state
        next_obs_g = step_out.outputs.guard.obs
        next_obs_b = step_out.outputs.bandit.obs
        reward_g = step_out.outputs.guard.reward
        reward_b = step_out.outputs.bandit.reward
        done = step_out.episode_done
        next_terminated = terminated | done

        # Freeze-on-done: hold previous state/obs/ps if already terminated entering this step.
        advance_es = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), es, next_es
        )
        advance_obs_g = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), obs_g, next_obs_g
        )
        advance_obs_b = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), obs_b, next_obs_b
        )
        advance_ps_g = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), ps_g, next_ps_g
        )
        advance_ps_b = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), ps_b, next_ps_b
        )

        out_reward_g = jnp.where(terminated, jnp.zeros_like(reward_g), reward_g)
        out_reward_b = jnp.where(terminated, jnp.zeros_like(reward_b), reward_b)

        logged = {
            "env_state": es,
            "guard_action": action_g,
            "bandit_action": action_b,
            "guard_reward": out_reward_g,
            "bandit_reward": out_reward_b,
            "episode_done": next_terminated,
            "guard_obs": obs_g,
            "bandit_obs": obs_b,
            "guard_ps": ps_g,
            "bandit_ps": ps_b,
        }
        return (
            advance_es, advance_ps_g, advance_ps_b,
            advance_obs_g, advance_obs_b, next_terminated,
        ), logged

    step_keys = jax.random.split(k_scan, n_steps)
    initial_obs_g = initial_outputs.guard.obs
    initial_obs_b = initial_outputs.bandit.obs
    _, stacked = jax.lax.scan(
        _step,
        (env_state, ps_g, ps_b, initial_obs_g, initial_obs_b, initial_terminated),
        step_keys,
    )

    return Trajectory(
        env_state=stacked["env_state"],
        sides=BySide(
            guard=SideTrajectory(
                obs=stacked["guard_obs"],
                action=stacked["guard_action"],
                reward=stacked["guard_reward"],
                done=stacked["episode_done"],
                policy_state=stacked["guard_ps"],
            ),
            bandit=SideTrajectory(
                obs=stacked["bandit_obs"],
                action=stacked["bandit_action"],
                reward=stacked["bandit_reward"],
                done=stacked["episode_done"],
                policy_state=stacked["bandit_ps"],
            ),
        ),
        episode_done=stacked["episode_done"],
        controlled_side=getattr(env.config, "controlled_side", Side.GUARD),
    )


def rollout_single_agent(
    view,
    controlled_policy: Callable,
    init_controlled_ps_fn: Callable,
    key: jax.Array,
    n_steps: int,
) -> Trajectory:
    """Convenience wrapper for SingleAgentView-style runs.

    Pulls the opponent's scripted policy from `view.opponent_policy` and
    threads only the controlled side's policy state externally.
    """
    if view.controlled_side is Side.GUARD:
        policies = BySide(guard=controlled_policy, bandit=view.opponent_policy)
        init_fns = BySide(guard=init_controlled_ps_fn, bandit=lambda c, s, k: None)
    else:
        policies = BySide(guard=view.opponent_policy, bandit=controlled_policy)
        init_fns = BySide(guard=lambda c, s, k: None, bandit=init_controlled_ps_fn)
    return rollout(view.env, policies, init_fns, key, n_steps)


def episode_mask(traj: Trajectory) -> jax.Array:
    """Boolean mask True up to and including the terminating step.

    Same semantics as Phase 0's episode_mask but reads from the new
    Trajectory shape's `episode_done` field.
    """
    done = traj.episode_done
    shifted = jnp.concatenate(
        [jnp.zeros_like(done[..., :1]), done[..., :-1]],
        axis=-1,
    )
    cumulative_any_before = jnp.cumsum(shifted, axis=-1) > 0
    return ~cumulative_any_before
