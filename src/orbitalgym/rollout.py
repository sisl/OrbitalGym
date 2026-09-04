"""``jax.lax.scan``-based rollouts for the symmetric core.

Two drivers:

- :func:`rollout` — both sides have policies; ``agent_view = obs`` (the
  flattened per-side observation). Used for memoryless / fully reactive
  policies.
- :func:`belief_rollout` — both sides additionally have belief initializers
  and updaters; ``agent_view = belief`` (the per-side
  :class:`orbitalgym.belief.base.Belief`, post-update). Used for
  state-aware policies (e.g. :class:`orbitalgym.policies.mcts.MCTSPolicy`)
  and for belief-conditioned controllers.

Both drivers thread (env_state, BySide[policy_state], BySide[agent_view],
terminated_flag) through ``lax.scan``. Each step:

  1. For each side, policy maps ``(ps_s, agent_view_s, key, t)`` →
     ``(action_s, ps_s')``.
  2. Bundle actions as ``Actions``; call ``env.step`` → ``StepOutput``.
  3. (belief_rollout only) Update each side's belief from the step's obs.
  4. Apply freeze-on-done logic per side (state/agent_view/ps held;
     reward zeroed).

Key-split note: the top-level split uses ``split(key, 3) → (k_reset,
k_init, k_scan)`` to preserve byte-identical numerical output with the
Phase-0 baseline fixture. ``k_init`` is further split into per-side init
keys (and additional belief-init keys for ``belief_rollout``). All current
observation functions and the dynamics step are deterministic (keys
discarded), so only ``k_reset`` affects numerical output via IC sampling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.belief.contact_aware import ContactAwareBelief
from orbitalgym.env.types import (
    Actions,
    BySide,
    Side,
    SideTrajectory,
    Trajectory,
)
from orbitalgym.groundstations.contacts import in_contact_now
from orbitalgym.observations.types import flatten_observations
from orbitalgym.registry import ActionComponentKey


def rollout(
    env,
    policies: BySide,  # BySide[Policy] — one policy per side
    init_policy_state_fns: BySide,  # BySide[Callable(config, env_state, key) -> ps]
    key: jax.Array,
    n_steps: int,
    *,
    initial_state: Any = None,
) -> Trajectory:
    """Symmetric obs-only rollout. Both sides driven by their own policies.

    ``agent_view`` passed to each policy is the flattened per-side observation.

    Returns a Trajectory with sides.guard / sides.bandit each carrying
    ``(T, N_side, ...)`` leading-axis arrays, plus ``episode_done (T,)`` latched.

    ``initial_state`` starts the episode from a stored ``EnvState`` (see
    ``orbitalgym.eval.bank``) instead of sampling.
    """
    k_reset, k_init, k_scan = jax.random.split(key, 3)
    k_init_g, k_init_b = jax.random.split(k_init, 2)
    if initial_state is None:
        env_state, initial_outputs = env.reset(k_reset)
    else:
        env_state, initial_outputs = env.reset_from_state(initial_state, k_reset)
    ps_g = init_policy_state_fns.guard(env.config, env_state, k_init_g)
    ps_b = init_policy_state_fns.bandit(env.config, env_state, k_init_b)
    initial_terminated = jnp.asarray(False)

    def _step(carry, step_key):
        es, ps_g, ps_b, view_g, view_b, terminated = carry
        k_act_g, k_act_b, k_env = jax.random.split(step_key, 3)
        action_g, next_ps_g = policies.guard(ps_g, view_g, k_act_g, es.t)
        action_b, next_ps_b = policies.bandit(ps_b, view_b, k_act_b, es.t)
        actions = Actions(sides=BySide(guard=action_g, bandit=action_b))
        step_out = env.step(k_env, es, actions)
        next_es = step_out.state
        next_view_g = flatten_observations(step_out.outputs.guard.obs)
        next_view_b = flatten_observations(step_out.outputs.bandit.obs)
        reward_g = step_out.outputs.guard.reward
        reward_b = step_out.outputs.bandit.reward
        done = step_out.episode_done
        next_terminated = terminated | done

        # Freeze-on-done: hold previous state/view/ps if already terminated entering this step.
        advance_es = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), es, next_es
        )
        advance_view_g = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), view_g, next_view_g
        )
        advance_view_b = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), view_b, next_view_b
        )
        advance_ps_g = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), ps_g, next_ps_g
        )
        advance_ps_b = jax.tree_util.tree_map(
            lambda cur, nxt: jnp.where(terminated, cur, nxt), ps_b, next_ps_b
        )

        out_reward_g = jnp.where(terminated, jnp.zeros_like(reward_g), reward_g)
        out_reward_b = jnp.where(terminated, jnp.zeros_like(reward_b), reward_b)

        applied_dv = step_out.info["applied_dv"]
        logged = {
            "env_state": es,
            "guard_action": action_g,
            "bandit_action": action_b,
            "guard_reward": out_reward_g,
            "bandit_reward": out_reward_b,
            "episode_done": next_terminated,
            "guard_obs": view_g,
            "bandit_obs": view_b,
            "guard_ps": ps_g,
            "bandit_ps": ps_b,
            "guard_applied_dv": applied_dv.guard,
            "bandit_applied_dv": applied_dv.bandit,
        }
        return (
            advance_es,
            advance_ps_g,
            advance_ps_b,
            advance_view_g,
            advance_view_b,
            next_terminated,
        ), logged

    step_keys = jax.random.split(k_scan, n_steps)
    initial_view_g = flatten_observations(initial_outputs.guard.obs)
    initial_view_b = flatten_observations(initial_outputs.bandit.obs)
    final_carry, stacked = jax.lax.scan(
        _step,
        (env_state, ps_g, ps_b, initial_view_g, initial_view_b, initial_terminated),
        step_keys,
    )

    return Trajectory(
        env_state=stacked["env_state"],
        final_state=final_carry[0],
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
        applied_dv=BySide(guard=stacked["guard_applied_dv"], bandit=stacked["bandit_applied_dv"]),
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

    Pulls the opponent's scripted policy from ``view.opponent_policy`` and
    threads only the controlled side's policy state externally.
    """
    if view.controlled_side is Side.GUARD:
        policies = BySide(guard=controlled_policy, bandit=view.opponent_policy)
        init_fns = BySide(guard=init_controlled_ps_fn, bandit=lambda c, s, k: None)
    else:
        policies = BySide(guard=view.opponent_policy, bandit=controlled_policy)
        init_fns = BySide(guard=lambda c, s, k: None, bandit=init_controlled_ps_fn)
    return rollout(view.env, policies, init_fns, key, n_steps)


def _require_impulsive_maneuver(config: Any, *, where: str) -> None:
    """Fail fast if ``dv`` is not on the assembled Command pytree.

    ``belief_rollout`` predicts belief means under control from the Δv the
    env actually imparted (``StepOutput.info["applied_dv"]``), which is
    nonzero only for a side carrying ``IMPULSIVE_MANEUVER``. Without this
    check, a comms-only side would silently predict free drift while its
    policy believed it was manoeuvring.
    """
    g = tuple(config.guard_action_components)
    b = tuple(config.bandit_action_components)
    has_g = ActionComponentKey.IMPULSIVE_MANEUVER in g
    has_b = ActionComponentKey.IMPULSIVE_MANEUVER in b
    if has_g and has_b:
        return
    raise ValueError(
        f"{where} requires IMPULSIVE_MANEUVER in guard_action_components and "
        f"bandit_action_components (it predicts belief from the Δv the env "
        f"applied for each side). Got: guard={g!r}, bandit={b!r}"
    )


def _any_opponent_visible(obs_channels: tuple, n_self: int) -> jax.Array:
    """Per own-vehicle bool: any opposing target visible in any channel.

    Each channel's ``visible`` has shape ``(n_self, n_total)``; columns
    ``n_self:`` are the opposing side. Returns shape ``(n_self,)``.
    """
    per_channel = [jnp.any(c.visible[:, n_self:], axis=-1) for c in obs_channels]
    return jnp.any(jnp.stack(per_channel, axis=0), axis=0)


def belief_rollout(
    env,
    policies: BySide,  # BySide[Policy]
    init_policy_state_fns: BySide,  # BySide[Callable(config, env_state, key) -> ps]
    belief_initializers: BySide,  # BySide[BeliefInitializer]
    belief_updaters: BySide,  # BySide[BeliefUpdater]
    key: jax.Array,
    n_steps: int,
    *,
    guard_ground_station_network: Any = None,
    bandit_ground_station_network: Any = None,
    team_sync_fns: Any = None,  # BySide[BeliefSyncFn | None] | None
    guard_link: Any = None,
    bandit_link: Any = None,
    initial_state: Any = None,
) -> tuple[Trajectory, BySide]:
    """Belief-aware rollout: agent_view = post-update Belief.

    Each tick:

      1. For each side, compute a per-tick contact mask of shape
         ``(n_side,)`` from the side's link predicate when given, else from
         its ground-station network. When neither is given the mask is
         all-False.
      2. Optionally fuse the side's belief across in-contact teammates via
         ``team_sync_fns.X(belief_X, contact_X)``.
      3. Wrap the (possibly synced) belief as a
         :class:`~orbitalgym.belief.contact_aware.ContactAwareBelief` and
         pass it to ``policies.X(ps_X, agent_view, key, t)``.
      4. ``env.step`` advances the world.
      5. Each side's ``belief_updater`` folds the new per-side observation
         channels into the synced belief (so team fusion persists into the
         next tick's prior). The control it predicts under is the Δv the
         env actually imparted, from ``StepOutput.info["applied_dv"]``, not
         the commanded Δv: a command clipped by thrust or an empty tank
         would otherwise leave the own-state belief drifting off truth.
      6. Freeze-on-done as in :func:`rollout`.

    The synced belief — not the wrapped ``ContactAwareBelief`` — is what
    flows forward in the carry: wrapping is a per-tick view, fusion is a
    persistent state update.

    Returns ``(Trajectory, BySide[belief_history])``. ``belief_history`` is a
    BySide whose leaves are belief leaves with a leading time axis — what
    feeds :class:`orbitalgym.viz.animation.RolloutScene`'s
    ``belief_history`` parameter for animated 2σ ellipsoids.

    ``initial_state`` starts the episode from a stored ``EnvState`` (see
    ``orbitalgym.eval.bank``) instead of sampling.
    """
    _require_impulsive_maneuver(env.config, where="belief_rollout")

    n_g = env.config.n_guards
    n_b = env.config.n_bandits
    # Applied Δv is stored padded to width 3; the belief dynamics take a
    # control of the truth frame's width.
    dv_dim = env.truth_frame.dim

    def _contact_mask(link: Any, network: Any, n_side: int, es: Any, side: Side) -> jax.Array:
        """Per-side link mask. A link predicate wins; else the network schedule; else all-False."""
        if link is not None:
            return link(es, side, es.t)
        if network is None:
            return jnp.zeros((n_side,), dtype=jnp.bool_)
        return jnp.broadcast_to(in_contact_now(network.schedule, es.t), (n_side,))

    k_reset, k_init, k_scan = jax.random.split(key, 3)
    k_init_g, k_init_b = jax.random.split(k_init, 2)
    k_b_init_g, k_b_init_b = jax.random.split(k_init_g, 2)

    if initial_state is None:
        env_state, _initial_outputs = env.reset(k_reset)
    else:
        env_state, _initial_outputs = env.reset_from_state(initial_state, k_reset)
    ps_g = init_policy_state_fns.guard(env.config, env_state, k_init_g)
    ps_b = init_policy_state_fns.bandit(env.config, env_state, k_init_b)
    belief_g = belief_initializers.guard(env_state, Side.GUARD, k_b_init_g)
    belief_b = belief_initializers.bandit(env_state, Side.BANDIT, k_b_init_b)
    initial_terminated = jnp.asarray(False)

    def _step(carry, step_key):
        es, ps_g, ps_b, belief_g, belief_b, terminated = carry
        k_act_g, k_act_b, k_env, k_g_obs, k_b_obs, k_g_upd, k_b_upd = jax.random.split(step_key, 7)

        # Per-tick contact masks (closure-static branch on the Python
        # network handles, traced lookup on `es.t`).
        contact_g = _contact_mask(guard_link, guard_ground_station_network, n_g, es, Side.GUARD)
        contact_b = _contact_mask(bandit_link, bandit_ground_station_network, n_b, es, Side.BANDIT)

        # Optional team-belief fusion. Runs BEFORE the policy call so the
        # policy sees the fused belief; the fused belief also replaces the
        # carry's belief for the post-step update so fusion persists.
        synced_belief_g = belief_g
        synced_belief_b = belief_b
        if team_sync_fns is not None:
            if team_sync_fns.guard is not None:
                synced_belief_g = team_sync_fns.guard(belief_g, contact_g)
            if team_sync_fns.bandit is not None:
                synced_belief_b = team_sync_fns.bandit(belief_b, contact_b)

        # Wrap as ContactAwareBelief for the policy call. Existing policies
        # that read only `.mean` are unaffected (the wrapper delegates);
        # contact-aware policies get `.contact` for plan-cache gating.
        view_g_for_policy = ContactAwareBelief(inner=synced_belief_g, contact=contact_g)
        view_b_for_policy = ContactAwareBelief(inner=synced_belief_b, contact=contact_b)

        # Policies receive the BELIEF as agent_view. This is the core
        # contract: belief is the agent's perception output, not the raw obs.
        action_g, next_ps_g = policies.guard(ps_g, view_g_for_policy, k_act_g, es.t)
        action_b, next_ps_b = policies.bandit(ps_b, view_b_for_policy, k_act_b, es.t)
        actions = Actions(sides=BySide(guard=action_g, bandit=action_b))
        step_out = env.step(k_env, es, actions)
        next_es = step_out.state

        guard_obs_channels = env.guard_observation_fn(
            next_es, actions, Side.GUARD, env.config, k_g_obs, next_es.t
        )
        bandit_obs_channels = env.bandit_observation_fn(
            next_es, actions, Side.BANDIT, env.config, k_b_obs, next_es.t
        )
        # Post-step update operates on the SYNCED belief (so team fusion
        # persists into the next tick's prior).
        applied_dv = step_out.info["applied_dv"]
        next_belief_g = belief_updaters.guard(
            synced_belief_g,
            guard_obs_channels,
            applied_dv.guard[:, :dv_dim],
            Side.GUARD,
            k_g_upd,
        )
        next_belief_b = belief_updaters.bandit(
            synced_belief_b,
            bandit_obs_channels,
            applied_dv.bandit[:, :dv_dim],
            Side.BANDIT,
            k_b_upd,
        )

        # The trajectory still records the flat obs for downstream tooling
        # (e.g. viz). It is computed from the step output, not passed to
        # the policy.
        view_g = flatten_observations(step_out.outputs.guard.obs)
        view_b = flatten_observations(step_out.outputs.bandit.obs)
        visible_g = _any_opponent_visible(step_out.outputs.guard.obs, n_g)
        visible_b = _any_opponent_visible(step_out.outputs.bandit.obs, n_b)
        reward_g = step_out.outputs.guard.reward
        reward_b = step_out.outputs.bandit.reward
        next_terminated = terminated | step_out.episode_done

        def freeze(cur, nxt):
            return jax.tree_util.tree_map(lambda c, n: jnp.where(terminated, c, n), cur, nxt)

        advance_es = freeze(es, next_es)
        advance_ps_g = freeze(ps_g, next_ps_g)
        advance_ps_b = freeze(ps_b, next_ps_b)
        # Freeze-on-done compares the synced (carried) belief vs the
        # post-update belief — keeps the synced one if already terminated.
        advance_belief_g = freeze(synced_belief_g, next_belief_g)
        advance_belief_b = freeze(synced_belief_b, next_belief_b)

        out_reward_g = jnp.where(terminated, jnp.zeros_like(reward_g), reward_g)
        out_reward_b = jnp.where(terminated, jnp.zeros_like(reward_b), reward_b)

        logged = {
            "env_state": es,
            "guard_action": action_g,
            "bandit_action": action_b,
            "guard_reward": out_reward_g,
            "bandit_reward": out_reward_b,
            "episode_done": next_terminated,
            "guard_obs": view_g,
            "bandit_obs": view_b,
            "guard_ps": ps_g,
            "bandit_ps": ps_b,
            "guard_belief": synced_belief_g,
            "bandit_belief": synced_belief_b,
            "guard_contact": contact_g,
            "bandit_contact": contact_b,
            "guard_visible": visible_g,
            "bandit_visible": visible_b,
            "guard_applied_dv": applied_dv.guard,
            "bandit_applied_dv": applied_dv.bandit,
        }
        return (
            advance_es,
            advance_ps_g,
            advance_ps_b,
            advance_belief_g,
            advance_belief_b,
            next_terminated,
        ), logged

    step_keys = jax.random.split(k_scan, n_steps)
    final_carry, stacked = jax.lax.scan(
        _step,
        (env_state, ps_g, ps_b, belief_g, belief_b, initial_terminated),
        step_keys,
    )

    traj = Trajectory(
        env_state=stacked["env_state"],
        final_state=final_carry[0],
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
        contact=BySide(guard=stacked["guard_contact"], bandit=stacked["bandit_contact"]),
        visible=BySide(guard=stacked["guard_visible"], bandit=stacked["bandit_visible"]),
        applied_dv=BySide(guard=stacked["guard_applied_dv"], bandit=stacked["bandit_applied_dv"]),
        controlled_side=getattr(env.config, "controlled_side", Side.GUARD),
    )
    belief_history = BySide(guard=stacked["guard_belief"], bandit=stacked["bandit_belief"])
    return traj, belief_history


def episode_mask(traj: Trajectory) -> jax.Array:
    """Boolean mask True up to and including the terminating step.

    Same semantics as Phase 0's ``episode_mask`` but reads from the new
    Trajectory shape's ``episode_done`` field.
    """
    done = traj.episode_done
    shifted = jnp.concatenate(
        [jnp.zeros_like(done[..., :1]), done[..., :-1]],
        axis=-1,
    )
    cumulative_any_before = jnp.cumsum(shifted, axis=-1) > 0
    return ~cumulative_any_before
