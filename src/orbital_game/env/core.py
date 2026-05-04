"""OrbitalGameEnv — symmetric multi-agent core.

Wires together: state classes (per scenario), dynamics, actuators,
observation fns, reward fn, termination fn, IC sampler. Exposes reset / step
with a symmetric API — both sides run identical machinery. Callers (typically
SingleAgentView) supply both sides' actions.

EnvState is defined here (not in canonical types) because its per-scenario pytree
structure depends on the assembled GuardState / BanditState classes.
"""

from __future__ import annotations

from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from orbital_game.actuators.impulsive import ImpulsiveActuator
from orbital_game.config import ScenarioConfig
from orbital_game.dynamics.hcw import hcw_rt_step, hcw_rtn_step
from orbital_game.env.types import Actions, BySide, Side, SideOutput, StepOutput
from orbital_game.reference_orbit import ReferenceOrbitState
from orbital_game.reference_orbit import mean_motion as _ref_mean_motion
from orbital_game.registry import ActuatorKey, DynamicsKey, StateComponentKey
from orbital_game.state.assemble import build_state_class
from orbital_game.state.components import (
    Attitude,
    BodyRates,
    Mass,
    Power,
    RTNState,
    RTState,
)
from orbital_game.state.layout import StateLayout

_COMP_LOOKUP = {
    StateComponentKey.RT: RTState,
    StateComponentKey.RTN: RTNState,
    StateComponentKey.MASS: Mass,
    StateComponentKey.POWER: Power,
    StateComponentKey.ATTITUDE: Attitude,
    StateComponentKey.BODY_RATES: BodyRates,
}

_DYN_LOOKUP = {
    DynamicsKey.HCW_RT: hcw_rt_step,
    DynamicsKey.HCW_RTN: hcw_rtn_step,
}


@flax.struct.dataclass
class _FleetParams:
    dry_mass_kg: jax.Array
    isp_s: jax.Array
    max_thrust_n: jax.Array
    mean_motion: jax.Array


def _build_per_side_params(spec, mean_motion: float) -> _FleetParams:
    return _FleetParams(
        dry_mass_kg=jnp.asarray(spec.dry_mass_kg),
        isp_s=jnp.asarray(spec.isp_s),
        max_thrust_n=jnp.asarray(spec.max_thrust_n),
        mean_motion=jnp.asarray(mean_motion),
    )


@flax.struct.dataclass
class EnvState:
    t: jax.Array  # scalar seconds since episode start
    step: jax.Array  # scalar int
    guards: Any  # assembled guard pytree
    bandits: Any  # assembled bandit pytree
    reference_orbit: ReferenceOrbitState
    # bool scalar — False iff IC rejection cap was exhausted at reset.
    # Defaults to True so legacy construction sites (and tests that build
    # EnvState without rejection sampling) get a sensible value.
    ic_valid: jax.Array = flax.struct.field(default_factory=lambda: jnp.asarray(True))


def _make_actuator(key: ActuatorKey, track_mass: bool):
    if key == ActuatorKey.IMPULSIVE:
        return ImpulsiveActuator(track_mass=track_mass)
    raise ValueError(f"Unknown actuator key: {key!r}")


def _get_dynamics_state(state) -> jax.Array:
    """Return the raw (n, 4) or (n, 6) array the dynamics expects.

    Queries the authoritative component tuple set by `build_state_class`
    rather than duck-typing via `hasattr`, so ordering is deterministic and
    a state with both `rt` and `rtn` would be an explicit error.
    """
    comps = state._orbital_game_components
    if RTNState in comps:
        return state.rtn
    if RTState in comps:
        return state.rt
    raise AttributeError("State has no dynamics component (RTState or RTNState)")


def _set_dynamics_state(state, new_state: jax.Array):
    comps = state._orbital_game_components
    if RTNState in comps:
        return state.replace(rtn=new_state)
    if RTState in comps:
        return state.replace(rt=new_state)
    raise AttributeError("State has no dynamics component (RTState or RTNState)")


def _apply_propellant(state, dp: jax.Array):
    if hasattr(state, "propellant_mass"):
        return state.replace(propellant_mass=jnp.maximum(state.propellant_mass - dp, 0.0))
    return state


class OrbitalGameEnv:
    """Composed environment. reset / step are pure functions under the hood."""

    def __init__(self, config: ScenarioConfig):
        self.config = config

        # Mean motion via vis-viva. Shared helper in reference_orbit.py — RelativeEllipse
        # uses the same so sampled ICs satisfy the dynamics' bounded-orbit condition.
        self.mean_motion = float(_ref_mean_motion(config.reference_orbit))

        # Build per-side state classes.
        guard_comps = [_COMP_LOOKUP[k] for k in config.guard_components]
        bandit_comps = [_COMP_LOOKUP[k] for k in config.bandit_components]
        self.guard_state_cls = build_state_class(guard_comps, config.n_guards, "GuardState")
        self.bandit_state_cls = build_state_class(bandit_comps, config.n_bandits, "BanditState")

        self.layout = StateLayout.build(
            guard_state_cls=self.guard_state_cls,
            bandit_state_cls=self.bandit_state_cls,
            n_guards=config.n_guards,
            n_bandits=config.n_bandits,
        )

        self.guard_params = _build_per_side_params(config.guard_params, self.mean_motion)
        self.bandit_params = _build_per_side_params(config.bandit_params, self.mean_motion)

        guard_track_mass = Mass in guard_comps
        bandit_track_mass = Mass in bandit_comps
        self.guard_actuator = _make_actuator(config.guard_actuator, guard_track_mass)
        self.bandit_actuator = _make_actuator(config.bandit_actuator, bandit_track_mass)

        self.truth_dynamics = _DYN_LOOKUP[config.truth_dynamics]
        self.planning_dynamics = _DYN_LOOKUP[config.planning_dynamics]

        # Typed-instance components come directly from the config (populated by
        # ScenarioConfig.__post_init__ defaults or overridden by the caller).
        self.guard_observation_fn = config.guard_observation_fn
        self.bandit_observation_fn = config.bandit_observation_fn
        self.reward_fn = config.reward_fn
        self.termination_fn = config.termination_fn
        # IC sampler comes from the scenario config — the user picks nominals,
        # sigmas, and (in the future) the sampler variant. The env just holds
        # the reference.
        self.ic_sampler = config.ic_sampler

    def reset(self, key: jax.Array) -> tuple[EnvState, BySide]:
        """Reset returns (env_state, BySide(guard=SideOutput, bandit=SideOutput))."""
        k_ic, k_obs_g, k_obs_b = jax.random.split(key, 3)
        guards, bandits, ok = self._reset_with_icspec(k_ic)
        state = EnvState(
            t=jnp.asarray(0.0),
            step=jnp.asarray(0),
            guards=guards,
            bandits=bandits,
            reference_orbit=self.config.reference_orbit,
            ic_valid=ok,
        )
        obs_g = self.guard_observation_fn(state, Side.GUARD, self.config, k_obs_g, state.t)
        obs_b = self.bandit_observation_fn(state, Side.BANDIT, self.config, k_obs_b, state.t)
        initial_done = jnp.asarray(False)
        initial_outputs = BySide(
            guard=SideOutput(obs=obs_g, reward=jnp.asarray(0.0), done=initial_done),
            bandit=SideOutput(obs=obs_b, reward=jnp.asarray(0.0), done=initial_done),
        )
        return state, initial_outputs

    def _reset_with_icspec(self, k_ic: jax.Array):
        """Rejection-sampling reset for ICSpec configurations.

        Runs `jax.lax.while_loop` capped at `ic_sampler.max_attempts`. If all
        validators pass, returns (guards, bandits, True). If the cap is exhausted,
        returns the last-drawn (guards, bandits) with ic_valid=False — non-fatal,
        so downstream code can surface infeasibility rates instead of crashing.
        """
        n_guards = self.config.n_guards
        n_bandits = self.config.n_bandits
        guard_components = self.config.guard_components
        bandit_components = self.config.bandit_components

        def _draw(i):
            k = jax.random.fold_in(k_ic, i)
            kg, kb = jax.random.split(k, 2)
            guards = self.ic_sampler.guard_sampler(
                self.config,
                kg,
                n_vehicles=n_guards,
                components=guard_components,
                class_name="GuardState",
            )
            bandits = self.ic_sampler.bandit_sampler(
                self.config,
                kb,
                n_vehicles=n_bandits,
                components=bandit_components,
                class_name="BanditState",
            )
            # Python-side branch on a static tuple length: safe under jit/vmap.
            if self.ic_sampler.validators:
                checks = jnp.stack(
                    [
                        jnp.asarray(v(self.config, guards, bandits))
                        for v in self.ic_sampler.validators
                    ]
                )
                ok = jnp.all(checks)
            else:
                ok = jnp.asarray(True)
            return guards, bandits, ok

        # Seed the carry with a concrete-shape draw so the while_loop has a
        # well-defined pytree structure.
        seed_guards, seed_bandits, seed_ok = _draw(jnp.asarray(0))
        init_carry = (jnp.asarray(1), seed_guards, seed_bandits, seed_ok)

        def _cond(carry):
            i, _, _, ok = carry
            return jnp.logical_and(jnp.logical_not(ok), i < self.ic_sampler.max_attempts)

        def _body(carry):
            i, _, _, _ = carry
            g, b, ok = _draw(i)
            return (i + 1, g, b, ok)

        _, guards, bandits, ok = jax.lax.while_loop(_cond, _body, init_carry)
        return guards, bandits, ok

    def step(self, key: jax.Array, state: EnvState, actions: Actions) -> StepOutput:
        """Symmetric step. Both sides run through identical machinery."""
        k_dyn, k_obs_g, k_obs_b = jax.random.split(key, 3)
        del k_dyn  # reserved for stochastic dynamics
        guard_action = actions.sides.guard
        bandit_action = actions.sides.bandit

        # Symmetric per-side dynamics step.
        applied_guard, dprop_guard = self.guard_actuator.apply(
            guard_action, state.guards, self.guard_params, self.config.dt
        )
        applied_bandit, dprop_bandit = self.bandit_actuator.apply(
            bandit_action, state.bandits, self.bandit_params, self.config.dt
        )
        new_guard_dyn = self.truth_dynamics(
            _get_dynamics_state(state.guards),
            applied_guard.dv,
            self.guard_params,
            self.config.dt,
        )
        new_bandit_dyn = self.truth_dynamics(
            _get_dynamics_state(state.bandits),
            applied_bandit.dv,
            self.bandit_params,
            self.config.dt,
        )
        next_guards = _set_dynamics_state(state.guards, new_guard_dyn)
        next_bandits = _set_dynamics_state(state.bandits, new_bandit_dyn)
        next_guards = _apply_propellant(next_guards, dprop_guard)
        next_bandits = _apply_propellant(next_bandits, dprop_bandit)

        next_state = state.replace(  # pyrefly: ignore[missing-attribute]
            t=state.t + self.config.dt,
            step=state.step + 1,
            guards=next_guards,
            bandits=next_bandits,
        )

        # Per-side observations + rewards.
        obs_g = self.guard_observation_fn(
            next_state, Side.GUARD, self.config, k_obs_g, next_state.t
        )
        obs_b = self.bandit_observation_fn(
            next_state, Side.BANDIT, self.config, k_obs_b, next_state.t
        )
        reward_g = self.reward_fn(state, actions, next_state, Side.GUARD, self.config, state.t)
        reward_b = self.reward_fn(state, actions, next_state, Side.BANDIT, self.config, state.t)

        episode_done = self.termination_fn(next_state, self.config, next_state.t)

        # Per-side done is the scalar broadcast (shape uniformity for adapters).
        return StepOutput(
            state=next_state,
            outputs=BySide(
                guard=SideOutput(obs=obs_g, reward=reward_g, done=episode_done),
                bandit=SideOutput(obs=obs_b, reward=reward_b, done=episode_done),
            ),
            episode_done=episode_done,
            info={},
        )
