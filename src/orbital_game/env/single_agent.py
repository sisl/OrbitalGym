"""SingleAgentView — projection wrapper for asymmetric play.

Reads `cfg.controlled_side` and the opposite side's scripted policy from
config; exposes a single-agent reset/step interface that internally runs
the scripted opponent and threads its policy state through.

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
from orbital_game.policies import ZeroControl


def _resolve_scripted(cfg, side: Side, n_vehicles: int, action_dim: int):
    """Pull the scripted policy for `side` from cfg, populating dims.

    Falls back to ZeroControl if the field doesn't exist (e.g. when this
    runs before Task 6 wires config fields). Same pattern Task 6's
    config-aware build will use.
    """
    field_name = f"{side.value}_scripted_policy"
    spec = getattr(cfg, field_name, None) or ZeroControl()
    return dataclasses.replace(spec, n_vehicles=n_vehicles, action_dim=action_dim)


def _action_dim_from_dynamics(dynamics_key) -> int:
    """Recover action_dim without importing the env's private helper.

    Mirrors env/core.py's _dyn_action_dim: HCW_RT → 2, HCW_RTN → 3.
    """
    from orbital_game.registry import DynamicsKey

    return 2 if dynamics_key is DynamicsKey.HCW_RT else 3


class SingleAgentView:
    """Projects the symmetric OrbitalGameEnv core to a single-agent view.

    Caller provides actions for the *controlled* side only. The opponent
    is driven by the scripted policy declared in `cfg.{side}_scripted_policy`
    (or ZeroControl if that field is not yet present on the config).
    """

    def __init__(self, env: OrbitalGameEnv):
        self.env = env
        self.config = env.config
        self.controlled_side: Side = getattr(env.config, "controlled_side", Side.GUARD)
        self.opponent_side: Side = self.controlled_side.opposite()
        opp_n = env.config.n_guards if self.opponent_side is Side.GUARD else env.config.n_bandits
        action_dim = _action_dim_from_dynamics(env.config.truth_dynamics)
        self.opponent_policy = _resolve_scripted(env.config, self.opponent_side, opp_n, action_dim)

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
        opp_obs_raw = opp_obs_fn(state, self.opponent_side, self.config, k_opp, state.t)
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
