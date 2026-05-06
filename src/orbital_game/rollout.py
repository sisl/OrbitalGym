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
from typing import Any

import jax
import jax.numpy as jnp

from orbital_game.env.types import (
    Actions,
    BySide,
    Side,
    SideTrajectory,
    Trajectory,
)
from orbital_game.observations.types import flatten_observations


def rollout(
    env,
    policies: BySide,  # BySide[Policy] — one policy per side
    init_policy_state_fns: BySide,  # BySide[Callable(config, env_state, key) -> ps]
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
        next_obs_g = flatten_observations(step_out.outputs.guard.obs)
        next_obs_b = flatten_observations(step_out.outputs.bandit.obs)
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
            advance_es,
            advance_ps_g,
            advance_ps_b,
            advance_obs_g,
            advance_obs_b,
            next_terminated,
        ), logged

    step_keys = jax.random.split(k_scan, n_steps)
    initial_obs_g = flatten_observations(initial_outputs.guard.obs)
    initial_obs_b = flatten_observations(initial_outputs.bandit.obs)
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


def rollout_with_planner(
    env,
    planner_side: Side,
    planner,
    opponent_policy: Callable,
    key: jax.Array,
    n_steps: int,
    init_planner_state: Any = None,
):
    """Manual Python rollout for one tree-search planner + one scripted opponent.

    Tree-search planners (MCTS, POMCPOW) have Python-level state and variable
    control flow that don't fit inside ``jax.lax.scan`` (so they can't be wired
    through the standard ``rollout``). This helper runs a Python loop over a
    jit-compiled ``env.step``, calling the planner's ``plan(state, key)`` each
    tick to get the planner-side action and threading the opponent's
    obs-based ``Policy`` for the other side.

    Returns a list-of-states + reward/action/done arrays — *not* a Trajectory
    pytree, because shapes vary with episode length under early termination.
    Callers that need a fixed-shape Trajectory can pad after the fact.

    Args:
        env: OrbitalGameEnv.
        planner_side: which side (Side.GUARD or Side.BANDIT) is driven by the planner.
        planner: any Planner-protocol object with ``plan(state, key) -> (cmd, planner_state)``.
        opponent_policy: a Policy-protocol callable for the other side, with
            ``command_cls`` and ``n_vehicles`` already injected.
        key: master PRNGKey.
        n_steps: max steps before forced termination.
        init_planner_state: initial planner-state passed to the first plan call.

    Returns:
        dict with keys ``states`` (list[EnvState]), ``rewards_planner``,
        ``rewards_opponent``, ``planner_actions``, ``opponent_actions``,
        ``episode_done`` (bool list), and ``terminated_step`` (int).
    """
    from orbital_game.observations.types import flatten_observations

    state, outputs = env.reset(jax.random.fold_in(key, 0))
    states = [state]
    rewards_planner: list[float] = []
    rewards_opp: list[float] = []
    planner_actions: list[Any] = []
    opp_actions: list[Any] = []
    dones: list[bool] = []
    planner_state = init_planner_state

    for step in range(n_steps):
        sub_key = jax.random.fold_in(key, step + 1)
        k_plan, k_opp, k_env = jax.random.split(sub_key, 3)

        planner_cmd, planner_state = planner.plan(state, k_plan)

        # Opponent sees its own observation derived from current state.
        opp_side = planner_side.opposite()
        opp_obs_fn = (
            env.guard_observation_fn if opp_side is Side.GUARD else env.bandit_observation_fn
        )
        identity = Actions(
            sides=BySide(
                guard=env.guard_command_cls.zeros(env.config.n_guards),
                bandit=env.bandit_command_cls.zeros(env.config.n_bandits),
            )
        )
        opp_obs_raw = opp_obs_fn(state, identity, opp_side, env.config, k_opp, state.t)
        opp_obs = flatten_observations(opp_obs_raw)
        opp_cmd, _ = opponent_policy(None, opp_obs, k_opp, state.t)

        if planner_side is Side.GUARD:
            actions = Actions(sides=BySide(guard=planner_cmd, bandit=opp_cmd))
        else:
            actions = Actions(sides=BySide(guard=opp_cmd, bandit=planner_cmd))

        step_out = env.step(k_env, state, actions)
        state = step_out.state
        states.append(state)
        rewards_planner.append(float(step_out.outputs.get(planner_side).reward))
        rewards_opp.append(float(step_out.outputs.get(opp_side).reward))
        planner_actions.append(planner_cmd)
        opp_actions.append(opp_cmd)
        dones.append(bool(step_out.episode_done))
        if step_out.episode_done:
            break

    return {
        "states": states,
        "rewards_planner": jnp.asarray(rewards_planner),
        "rewards_opponent": jnp.asarray(rewards_opp),
        "planner_actions": planner_actions,
        "opponent_actions": opp_actions,
        "episode_done": dones,
        "terminated_step": len(rewards_planner),
        "final_planner_state": planner_state,
    }


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
