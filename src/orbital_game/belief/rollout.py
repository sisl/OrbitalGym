"""BeliefRollout — drives env.step + per-side belief updates.

Helper for tests, notebooks, ad-hoc scripts. Production planners are expected
to thread belief themselves; this is not load-bearing on the env loop.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from orbital_game.env.types import Actions, BySide, Side, SideTrajectory, Trajectory
from orbital_game.observations.types import flatten_observations
from orbital_game.registry import ActionComponentKey


def _require_impulsive_maneuver(config: Any, *, where: str) -> None:
    """Fail fast at construction if `dv` is not on the assembled Command pytree.

    `BeliefRollout` and (downstream) belief updaters read
    `actions.sides.X.dv` to predict belief means under control, so both sides'
    action components must include `IMPULSIVE_MANEUVER`. Without this check,
    a comms-only side would raise an opaque AttributeError deep inside the
    rollout's lax.scan body.
    """
    g = tuple(config.guard_action_components)
    b = tuple(config.bandit_action_components)
    has_g = ActionComponentKey.IMPULSIVE_MANEUVER in g
    has_b = ActionComponentKey.IMPULSIVE_MANEUVER in b
    if has_g and has_b:
        return
    raise ValueError(
        f"{where} requires IMPULSIVE_MANEUVER in guard_action_components and "
        f"bandit_action_components (it reads action.sides.X.dv to predict "
        f"belief). Got: guard={g!r}, bandit={b!r}"
    )


class BeliefRollout:
    """Pairs an OrbitalGameEnv with per-side belief initializers + updaters."""

    def __init__(
        self,
        env: Any,
        guard_belief_initializer: Any,
        guard_belief_updater: Any,
        bandit_belief_initializer: Any,
        bandit_belief_updater: Any,
    ):
        _require_impulsive_maneuver(env.config, where="BeliefRollout")
        self.env = env
        self.guard_belief_initializer = guard_belief_initializer
        self.guard_belief_updater = guard_belief_updater
        self.bandit_belief_initializer = bandit_belief_initializer
        self.bandit_belief_updater = bandit_belief_updater

    def reset(self, key: jax.Array):
        """Returns (env_state, BySide(guard=KFBelief, bandit=KFBelief))."""
        k_env, k_g_init, k_b_init = jax.random.split(key, 3)
        state, _outs = self.env.reset(k_env)
        beliefs = BySide(
            guard=self.guard_belief_initializer(state, Side.GUARD, k_g_init),
            bandit=self.bandit_belief_initializer(state, Side.BANDIT, k_b_init),
        )
        return state, beliefs

    def step(self, key: jax.Array, state, beliefs, actions: Actions):
        """One env step + per-side belief update.

        Returns (next_state, next_beliefs, step_output).
        """
        k_env, k_g_obs, k_b_obs, k_g_upd, k_b_upd = jax.random.split(key, 5)
        step_out = self.env.step(k_env, state, actions)
        guard_obs = self.env.guard_observation_fn(
            step_out.state, actions, Side.GUARD, self.env.config, k_g_obs, step_out.state.t
        )
        bandit_obs = self.env.bandit_observation_fn(
            step_out.state, actions, Side.BANDIT, self.env.config, k_b_obs, step_out.state.t
        )
        next_beliefs = BySide(
            guard=self.guard_belief_updater(
                beliefs.guard, guard_obs, actions.sides.guard.dv, Side.GUARD, k_g_upd
            ),
            bandit=self.bandit_belief_updater(
                beliefs.bandit, bandit_obs, actions.sides.bandit.dv, Side.BANDIT, k_b_upd
            ),
        )
        return step_out.state, next_beliefs, step_out


def run_belief_rollout(
    belief_rollout: BeliefRollout,
    policies: BySide,
    init_policy_state_fns: BySide,
    key: jax.Array,
    n_steps: int,
) -> tuple[Trajectory, BySide]:
    """``jax.lax.scan`` over a ``BeliefRollout`` to produce a full Trajectory
    plus a time-stacked per-side belief history.

    Mirrors :func:`orbital_game.rollout.rollout` (same freeze-on-done logic,
    same ``Trajectory`` shape) and additionally returns ``BySide`` whose
    leaves are belief leaves with a leading time axis. The belief history
    is what feeds :class:`orbital_game.viz.animation.RolloutScene`'s
    ``belief_history`` parameter for animated 2σ ellipsoids.
    """
    env = belief_rollout.env

    k_reset, k_init, k_scan = jax.random.split(key, 3)
    k_init_g, k_init_b = jax.random.split(k_init, 2)
    k_b_init_g, k_b_init_b = jax.random.split(k_init_g, 2)

    env_state, initial_outputs = env.reset(k_reset)
    ps_g = init_policy_state_fns.guard(env.config, env_state, k_init_g)
    ps_b = init_policy_state_fns.bandit(env.config, env_state, k_init_b)
    belief_g = belief_rollout.guard_belief_initializer(env_state, Side.GUARD, k_b_init_g)
    belief_b = belief_rollout.bandit_belief_initializer(env_state, Side.BANDIT, k_b_init_b)
    initial_terminated = jnp.asarray(False)

    def _step(carry, step_key):
        es, ps_g, ps_b, obs_g, obs_b, belief_g, belief_b, terminated = carry
        k_act_g, k_act_b, k_env, k_g_obs, k_b_obs, k_g_upd, k_b_upd = jax.random.split(step_key, 7)
        action_g, next_ps_g = policies.guard(ps_g, obs_g, k_act_g, es.t)
        action_b, next_ps_b = policies.bandit(ps_b, obs_b, k_act_b, es.t)
        actions = Actions(sides=BySide(guard=action_g, bandit=action_b))
        step_out = env.step(k_env, es, actions)
        next_es = step_out.state

        guard_obs_channels = env.guard_observation_fn(
            next_es, actions, Side.GUARD, env.config, k_g_obs, next_es.t
        )
        bandit_obs_channels = env.bandit_observation_fn(
            next_es, actions, Side.BANDIT, env.config, k_b_obs, next_es.t
        )
        next_belief_g = belief_rollout.guard_belief_updater(
            belief_g, guard_obs_channels, action_g.dv, Side.GUARD, k_g_upd
        )
        next_belief_b = belief_rollout.bandit_belief_updater(
            belief_b, bandit_obs_channels, action_b.dv, Side.BANDIT, k_b_upd
        )

        next_obs_g = flatten_observations(step_out.outputs.guard.obs)
        next_obs_b = flatten_observations(step_out.outputs.bandit.obs)
        reward_g = step_out.outputs.guard.reward
        reward_b = step_out.outputs.bandit.reward
        next_terminated = terminated | step_out.episode_done

        # Freeze-on-done: hold previous state/obs/ps/belief if already terminated.
        def freeze(cur, nxt):
            return jax.tree_util.tree_map(lambda c, n: jnp.where(terminated, c, n), cur, nxt)

        advance_es = freeze(es, next_es)
        advance_obs_g = freeze(obs_g, next_obs_g)
        advance_obs_b = freeze(obs_b, next_obs_b)
        advance_ps_g = freeze(ps_g, next_ps_g)
        advance_ps_b = freeze(ps_b, next_ps_b)
        advance_belief_g = freeze(belief_g, next_belief_g)
        advance_belief_b = freeze(belief_b, next_belief_b)

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
            "guard_belief": belief_g,
            "bandit_belief": belief_b,
        }
        return (
            advance_es,
            advance_ps_g,
            advance_ps_b,
            advance_obs_g,
            advance_obs_b,
            advance_belief_g,
            advance_belief_b,
            next_terminated,
        ), logged

    step_keys = jax.random.split(k_scan, n_steps)
    initial_obs_g = flatten_observations(initial_outputs.guard.obs)
    initial_obs_b = flatten_observations(initial_outputs.bandit.obs)
    _, stacked = jax.lax.scan(
        _step,
        (
            env_state,
            ps_g,
            ps_b,
            initial_obs_g,
            initial_obs_b,
            belief_g,
            belief_b,
            initial_terminated,
        ),
        step_keys,
    )

    traj = Trajectory(
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
    belief_history = BySide(guard=stacked["guard_belief"], bandit=stacked["bandit_belief"])
    return traj, belief_history
