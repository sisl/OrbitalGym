"""OrbitalGameEnv — the composed environment.

Wires together: state classes (per scenario), dynamics, actuators, intruder policy,
observation fn, reward fn, termination fn, IC sampler. Exposes reset / step with
a lightweight API. Strict gymnax Environment inheritance is deferred (see spec
§Known gaps).

EnvState is defined here (not in canonical types) because its per-scenario pytree
structure depends on the assembled DefenderState / IntruderState classes.
"""

from __future__ import annotations

from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from orbital_game.actuators.impulsive import ImpulsiveActuator
from orbital_game.config import ScenarioConfig
from orbital_game.dynamics.hcw import hcw_rt_step, hcw_rtn_step
from orbital_game.hva import HVAState
from orbital_game.hva import mean_motion as _hva_mean_motion
from orbital_game.observations.reference import FullObservation
from orbital_game.policies.intruder import ZeroControlIntruder
from orbital_game.registry import ActuatorKey, DynamicsKey, StateComponentKey
from orbital_game.rewards.reference import DistanceToHVA
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
from orbital_game.termination.reference import MaxStepsOrBreach

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
    defenders: Any  # assembled defender pytree
    intruders: Any  # assembled intruder pytree
    hva: HVAState
    # bool scalar — False iff IC rejection cap was exhausted at reset.
    # Defaults to True so legacy construction sites (and tests that build
    # EnvState without rejection sampling) get a sensible value.
    ic_valid: jax.Array = flax.struct.field(default_factory=lambda: jnp.asarray(True))


def _dyn_action_dim(dyn_key: DynamicsKey) -> int:
    return 2 if dyn_key == DynamicsKey.HCW_RT else 3


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

        # Mean motion via vis-viva. Shared helper in hva.py — RelativeEllipse
        # uses the same so sampled ICs satisfy the dynamics' bounded-orbit condition.
        self.mean_motion = float(_hva_mean_motion(config.hva))

        # Build per-side state classes.
        def_comps = [_COMP_LOOKUP[k] for k in config.defender_components]
        int_comps = [_COMP_LOOKUP[k] for k in config.intruder_components]
        self.defender_state_cls = build_state_class(def_comps, config.n_defenders, "DefenderState")
        self.intruder_state_cls = build_state_class(int_comps, config.n_intruders, "IntruderState")

        self.layout = StateLayout.build(
            defender_state_cls=self.defender_state_cls,
            intruder_state_cls=self.intruder_state_cls,
            n_defenders=config.n_defenders,
            n_intruders=config.n_intruders,
        )

        self.defender_params = _build_per_side_params(config.defender_params, self.mean_motion)
        self.intruder_params = _build_per_side_params(config.intruder_params, self.mean_motion)

        # For bootstrap, construct reference pluggables directly. Registry-driven
        # resolution is a follow-up (noted in spec §Known gaps).
        def_track_mass = Mass in def_comps
        int_track_mass = Mass in int_comps
        self.defender_actuator = _make_actuator(config.defender_actuator, def_track_mass)
        self.intruder_actuator = _make_actuator(config.intruder_actuator, int_track_mass)

        self.truth_dynamics = _DYN_LOOKUP[config.truth_dynamics]
        self.planning_dynamics = _DYN_LOOKUP[config.planning_dynamics]

        self.intruder_policy = ZeroControlIntruder(
            n_intruders=config.n_intruders,
            action_dim=_dyn_action_dim(config.truth_dynamics),
        )
        # Separate observation functions per side. Currently both resolve to the
        # same FullObservation reference, but they are independent objects so a
        # realistic scenario can swap defender-side for a masked/noisy sensor
        # model while leaving the intruder-side omniscient (or vice versa).
        self.defender_observation_fn = FullObservation(layout=self.layout)
        self.intruder_observation_fn = FullObservation(layout=self.layout)
        self.reward_fn = DistanceToHVA()
        self.termination_fn = MaxStepsOrBreach(max_steps=config.max_steps, breach_distance_m=10.0)
        # IC sampler comes from the scenario config — the user picks nominals,
        # sigmas, and (in the future) the sampler variant. The env just holds
        # the reference.
        self.ic_sampler = config.ic_sampler

    def reset(self, key: jax.Array) -> tuple[EnvState, jax.Array]:
        # Split so IC sampling and the initial observation use independent keys.
        # Currently the reference pluggables ignore the observation key, but
        # noisy observation models will consume it; keep them uncoupled from IC.
        k_ic, k_obs = jax.random.split(key, 2)
        defs, ints, ok = self._reset_with_icspec(k_ic)
        state = EnvState(
            t=jnp.asarray(0.0),
            step=jnp.asarray(0),
            defenders=defs,
            intruders=ints,
            hva=self.config.hva,
            ic_valid=ok,
        )
        obs = self.defender_observation_fn(state, None, k_obs, state.t)
        return state, obs

    def _reset_with_icspec(self, k_ic: jax.Array):
        """Rejection-sampling reset for ICSpec configurations.

        Runs `jax.lax.while_loop` capped at `ic_sampler.max_attempts`. If all
        validators pass, returns (defs, ints, True). If the cap is exhausted,
        returns the last-drawn (defs, ints) with ic_valid=False — non-fatal,
        so downstream code can surface infeasibility rates instead of crashing.
        """
        n_def = self.config.n_defenders
        n_int = self.config.n_intruders
        def_components = self.config.defender_components
        int_components = self.config.intruder_components

        def _draw(i):
            k = jax.random.fold_in(k_ic, i)
            kd, ki = jax.random.split(k, 2)
            defs = self.ic_sampler.defender_sampler(
                self.config, kd,
                n_vehicles=n_def, components=def_components, class_name="DefenderState",
            )
            ints = self.ic_sampler.intruder_sampler(
                self.config, ki,
                n_vehicles=n_int, components=int_components, class_name="IntruderState",
            )
            # Python-side branch on a static tuple length: safe under jit/vmap.
            if self.ic_sampler.validators:
                checks = jnp.stack([
                    jnp.asarray(v(self.config, defs, ints))
                    for v in self.ic_sampler.validators
                ])
                ok = jnp.all(checks)
            else:
                ok = jnp.asarray(True)
            return defs, ints, ok

        # Seed the carry with a concrete-shape draw so the while_loop has a
        # well-defined pytree structure.
        seed_defs, seed_ints, seed_ok = _draw(jnp.asarray(0))
        init_carry = (jnp.asarray(1), seed_defs, seed_ints, seed_ok)

        def _cond(carry):
            i, _, _, ok = carry
            return jnp.logical_and(jnp.logical_not(ok), i < self.ic_sampler.max_attempts)

        def _body(carry):
            i, _, _, _ = carry
            d, n, ok = _draw(i)
            return (i + 1, d, n, ok)

        _, defs, ints, ok = jax.lax.while_loop(_cond, _body, init_carry)
        return defs, ints, ok

    def step(self, key: jax.Array, state: EnvState, action: jax.Array):
        # Separate keys for intruder obs+action, dynamics (if stochastic), and
        # terminal defender obs. Reference implementations ignore the keys, but
        # stochastic observation / dynamics models would produce correlated
        # noise if the keys were shared.
        k_intr, k_dyn, k_obs = jax.random.split(key, 3)

        # Intruder observes + acts via its own (independent) observation fn.
        obs_intr = self.intruder_observation_fn(state, None, k_intr, state.t)
        intr_action = self.intruder_policy(obs_intr, k_intr, state.t)

        # Actuators: command -> applied control + propellant delta (zeros if track_mass=False).
        applied_def, delta_propellant_def = self.defender_actuator.apply(
            action, state.defenders, self.defender_params, self.config.dt
        )
        applied_int, delta_propellant_int = self.intruder_actuator.apply(
            intr_action, state.intruders, self.intruder_params, self.config.dt
        )

        # Truth dynamics on the raw rtn/rt state array.
        new_def_state = self.truth_dynamics(
            _get_dynamics_state(state.defenders),
            applied_def.dv,
            self.defender_params,
            self.config.dt,
        )
        new_int_state = self.truth_dynamics(
            _get_dynamics_state(state.intruders),
            applied_int.dv,
            self.intruder_params,
            self.config.dt,
        )
        del k_dyn  # reserved for stochastic dynamics (unused by HCW)
        next_def = _set_dynamics_state(state.defenders, new_def_state)
        next_int = _set_dynamics_state(state.intruders, new_int_state)

        # Apply propellant deltas if Mass is tracked.
        next_def = _apply_propellant(next_def, delta_propellant_def)
        next_int = _apply_propellant(next_int, delta_propellant_int)

        next_state = state.replace(  # pyrefly: ignore[missing-attribute]
            t=state.t + self.config.dt,
            step=state.step + 1,
            defenders=next_def,
            intruders=next_int,
        )

        reward = self.reward_fn(state, action, next_state, None, state.t)
        done = self.termination_fn(next_state, None, next_state.t)
        obs = self.defender_observation_fn(next_state, None, k_obs, next_state.t)
        return next_state, obs, reward, done, {}
