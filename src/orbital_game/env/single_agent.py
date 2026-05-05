"""SingleAgentView — projection wrapper for asymmetric play.

Reads `cfg.controlled_side` and the opposite side's policy from config;
exposes a single-agent reset/step interface that internally runs the
opponent and threads its policy state through.

This is the only place the asymmetric controlled/opponent split exists.
The symmetric core (env/core.py) knows nothing about it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax

from orbital_game.env.core import EnvState, OrbitalGameEnv
from orbital_game.env.types import Actions, BySide, Side
from orbital_game.observations.types import flatten_observations


def _resolve_policy(cfg, side: Side, n_vehicles: int, command_cls):
    """Pull the policy for `side` from cfg, populating dims + Command class.

    `cfg.{side}_policy` is guaranteed to be a Policy instance by
    `ScenarioConfig.__post_init__` (defaults to `ZeroControl()` if the
    builder didn't supply one).
    """
    field_name = f"{side.value}_policy"
    spec = getattr(cfg, field_name)
    return dataclasses.replace(spec, n_vehicles=n_vehicles, command_cls=command_cls)


class SingleAgentView:
    """Projects the symmetric OrbitalGameEnv core to a single-agent view.

    Caller provides actions for the *controlled* side only. The opponent
    is driven by the policy declared in `cfg.{side}_policy`, which
    `ScenarioConfig.__post_init__` defaults to `ZeroControl()` when the
    builder leaves it unset.
    """

    def __init__(self, env: OrbitalGameEnv):
        self.env = env
        self.config = env.config
        self.controlled_side: Side = env.config.controlled_side
        self.opponent_side: Side = self.controlled_side.opposite()
        opp_n = env.config.n_guards if self.opponent_side is Side.GUARD else env.config.n_bandits
        opp_command_cls = (
            env.guard_command_cls if self.opponent_side is Side.GUARD else env.bandit_command_cls
        )
        self.opponent_policy = _resolve_policy(
            env.config, self.opponent_side, opp_n, opp_command_cls
        )

    def reset(self, key: jax.Array):
        """Returns (env_state, obs_controlled, opponent_policy_state)."""
        state, outputs = self.env.reset(key)
        obs_controlled = flatten_observations(outputs.get(self.controlled_side).obs)
        # ZeroControl is stateless; richer policies overload init_policy_state.
        opp_policy_state = None
        return state, obs_controlled, opp_policy_state

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        controlled_action: jax.Array,
        opp_policy_state: Any,
    ):
        """Returns (next_state, obs_controlled, reward_controlled, episode_done, opp_ps', info)."""
        k_opp, k_env = jax.random.split(key, 2)
        # Get opponent's observation from current state to feed its policy.
        opp_obs_fn = (
            self.env.bandit_observation_fn
            if self.opponent_side is Side.BANDIT
            else self.env.guard_observation_fn
        )
        identity_actions = Actions(
            sides=BySide(
                guard=self.env.guard_command_cls.zeros(self.config.n_guards),
                bandit=self.env.bandit_command_cls.zeros(self.config.n_bandits),
            )
        )
        opp_obs_raw = opp_obs_fn(
            state, identity_actions, self.opponent_side, self.config, k_opp, state.t
        )
        opp_obs = flatten_observations(opp_obs_raw)
        opp_action, next_opp_ps = self.opponent_policy(opp_policy_state, opp_obs, k_opp, state.t)
        if self.controlled_side is Side.GUARD:
            actions = Actions(sides=BySide(guard=controlled_action, bandit=opp_action))
        else:
            actions = Actions(sides=BySide(guard=opp_action, bandit=controlled_action))
        step_out = self.env.step(k_env, state, actions)
        controlled_output = step_out.outputs.get(self.controlled_side)
        return (
            step_out.state,
            flatten_observations(controlled_output.obs),
            controlled_output.reward,
            step_out.episode_done,
            next_opp_ps,
            step_out.info,
        )
