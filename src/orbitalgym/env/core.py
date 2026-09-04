"""OrbitalGymEnv — symmetric multi-agent core.

Wires together: state classes (per scenario), dynamics, action components,
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

from orbitalgym.actions.assemble import build_command_class
from orbitalgym.actions.components import AttitudeControl, Communicate, ImpulsiveManeuver, PointAt
from orbitalgym.config import ScenarioConfig
from orbitalgym.dynamics.hcw import hcw_rt_step, hcw_rtn_step
from orbitalgym.dynamics.j2 import j2_eci_step
from orbitalgym.dynamics.keplerian import KeplerianEciDynamics
from orbitalgym.env.types import Actions, BySide, Side, SideOutput, StepOutput
from orbitalgym.frames.conversions import convert_state
from orbitalgym.reference_orbit import ReferenceOrbitState
from orbitalgym.reference_orbit import mean_motion as _ref_mean_motion
from orbitalgym.registry import (
    ActionComponentKey,
    DynamicsKey,
    Frame,
    StateComponentKey,
    resolve,
)
from orbitalgym.state.assemble import build_state_class
from orbitalgym.state.components import (
    AppliedDV,
    AppliedTorque,
    Attitude,
    BodyRates,
    ECIState,
    Mass,
    Power,
    RTNState,
    RTState,
)
from orbitalgym.state.layout import StateLayout

_COMP_LOOKUP = {
    StateComponentKey.RT: RTState,
    StateComponentKey.RTN: RTNState,
    StateComponentKey.ECI: ECIState,
    StateComponentKey.MASS: Mass,
    StateComponentKey.POWER: Power,
    StateComponentKey.ATTITUDE: Attitude,
    StateComponentKey.BODY_RATES: BodyRates,
    StateComponentKey.APPLIED_DV: AppliedDV,
    StateComponentKey.APPLIED_TORQUE: AppliedTorque,
}

# Mirror of `_COMP_LOOKUP` for action components. Resolves an
# ``ActionComponentKey`` to the dataclass that backs it. The env builds a
# concrete instance per side from this class, threading scenario-derived knobs
# (truth_dynamics, frames, mass tracking) into ``ImpulsiveManeuver`` and
# constructing ``Communicate`` directly.
_ACTION_COMP_LOOKUP: dict[ActionComponentKey, type] = {
    ActionComponentKey.IMPULSIVE_MANEUVER: ImpulsiveManeuver,
    ActionComponentKey.COMMUNICATE: Communicate,
    ActionComponentKey.ATTITUDE_CONTROL: AttitudeControl,
    ActionComponentKey.POINT_AT: PointAt,
}

_DYN_LOOKUP = {
    DynamicsKey.HCW_RT: hcw_rt_step,
    DynamicsKey.HCW_RTN: hcw_rtn_step,
    # Typed-instance dynamics: the registry holds the class; we instantiate
    # with defaults at env-build time. Per-scenario epoch is handled by
    # ScenarioConfig.__post_init__, which constructs an explicit instance
    # for `reference_orbit_dynamics`.
    DynamicsKey.KEPLERIAN_ECI: KeplerianEciDynamics,
    DynamicsKey.J2_ECI: j2_eci_step,
}


def _resolve_dynamics_callable(spec):
    """Return the callable for a dynamics-role spec.

    ``spec`` is one of:
    - a ``DynamicsKey`` looked up in ``_DYN_LOOKUP``. The looked-up entry may
      be a step *function* (HCW_RT, HCW_RTN, J2_ECI) or a typed-instance
      *class* (KEPLERIAN_ECI). Classes are instantiated with defaults here.
    - an already-constructed typed-instance (KeplerianEciDynamics,
      AstrojaxOrbitDynamics, ...) which is itself callable.
    """
    if isinstance(spec, DynamicsKey):
        thing = _DYN_LOOKUP[spec]
        if isinstance(thing, type):
            return thing()
        return thing
    return spec


# Per-frame canonical field name on the assembled per-side state pytree. The
# truth dynamics writes here; derived views are populated by converting from it.
_FRAME_TO_FIELD: dict[Frame, str] = {
    Frame.RT: "rt",
    Frame.RTN: "rtn",
    Frame.ECI: "eci",
}

# Inverse of the registry's frame-component mapping, restricted to spatial-state
# components. Mass/Power/Attitude/BodyRates have no frame and are skipped.
_COMPONENT_TO_FRAME: dict[StateComponentKey, Frame] = {
    StateComponentKey.RT: Frame.RT,
    StateComponentKey.RTN: Frame.RTN,
    StateComponentKey.ECI: Frame.ECI,
}


def _truth_field(truth_frame: Frame) -> str:
    return _FRAME_TO_FIELD[truth_frame]


def _materialize_derived_views(
    side_state,
    truth_field: str,
    truth_frame: Frame,
    extended_frames: tuple[Frame, ...],
    ref_eci6: jax.Array,
):
    """Populate every non-truth frame field on the per-side state by converting
    from the canonical truth-frame array. Returns the side state unchanged when
    `extended_frames` contains only the truth frame (e.g. HCW-only scenarios)."""
    truth_arr = getattr(side_state, truth_field)
    replacements: dict[str, jax.Array] = {}
    for f in extended_frames:
        if f is truth_frame:
            continue
        replacements[_FRAME_TO_FIELD[f]] = convert_state(truth_arr, truth_frame, f, ref_eci6)
    return side_state.replace(**replacements) if replacements else side_state


@flax.struct.dataclass
class _FleetParams:
    dry_mass_kg: jax.Array
    isp_s: jax.Array
    max_thrust_n: jax.Array
    mean_motion: jax.Array
    slew_rate_rad_s: jax.Array


def _build_per_side_params(spec, mean_motion: float) -> _FleetParams:
    return _FleetParams(
        dry_mass_kg=jnp.asarray(spec.dry_mass_kg),
        isp_s=jnp.asarray(spec.isp_s),
        max_thrust_n=jnp.asarray(spec.max_thrust_n),
        mean_motion=jnp.asarray(mean_motion),
        slew_rate_rad_s=jnp.asarray(spec.slew_rate_rad_s),
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


class OrbitalGymEnv:
    """Composed environment. reset / step are pure functions under the hood."""

    def __init__(self, config: ScenarioConfig):
        self.config = config

        # Mean motion via vis-viva. Shared helper in reference_orbit.py — RelativeEllipse
        # uses the same so sampled ICs satisfy the dynamics' bounded-orbit condition.
        self.mean_motion = float(_ref_mean_motion(config.reference_orbit))

        # Build per-side state classes from the auto-extended component tuples
        # so derived-frame views are present alongside the user-specified ones.
        guard_comps = [_COMP_LOOKUP[k] for k in config.guard_components_extended]
        bandit_comps = [_COMP_LOOKUP[k] for k in config.bandit_components_extended]
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

        self.truth_dynamics = _resolve_dynamics_callable(config.truth_dynamics)
        self.policy_dynamics = _resolve_dynamics_callable(config.policy_dynamics)
        self.belief_dynamics = _resolve_dynamics_callable(config.belief_dynamics_resolved)
        self.reference_orbit_dynamics = _resolve_dynamics_callable(
            config.reference_orbit_dynamics_resolved
        )

        # Attitude dynamics + per-side params (None when not configured).
        self.attitude_dynamics: Any | None
        if config.attitude_dynamics_key is not None:
            self.attitude_dynamics = resolve(config.attitude_dynamics_key)
            self.guard_attitude_params = config.guard_attitude_params
            self.bandit_attitude_params = config.bandit_attitude_params
        else:
            self.attitude_dynamics = None
            self.guard_attitude_params = None
            self.bandit_attitude_params = None

        # Per-side booleans: True iff that side has ATTITUDE state components.
        # Used in step() to guard attitude propagation on each side independently,
        # so asymmetric configurations (only one side has ATTITUDE+BODY_RATES)
        # do not raise AttributeError on the other side's .quat/.omega access.
        self.guard_has_attitude = StateComponentKey.ATTITUDE in config.guard_components
        self.bandit_has_attitude = StateComponentKey.ATTITUDE in config.bandit_components

        # Precompute the truth frame and the per-side spatial-frame set so step()
        # can look up canonical / derived fields without re-resolving registry
        # metadata every call. Mass/Power/etc. have no frame and are filtered out.
        self.truth_frame: Frame = self.truth_dynamics.frame

        # action_frame is resolved to a concrete Frame in ScenarioConfig.__post_init__;
        # the | None on the dataclass field is only for the user-facing API.
        assert config.action_frame is not None
        self.action_frame: Frame = config.action_frame
        self.guard_extended_frames: tuple[Frame, ...] = tuple(
            _COMPONENT_TO_FRAME[c]
            for c in config.guard_components_extended
            if c in _COMPONENT_TO_FRAME
        )
        self.bandit_extended_frames: tuple[Frame, ...] = tuple(
            _COMPONENT_TO_FRAME[c]
            for c in config.bandit_components_extended
            if c in _COMPONENT_TO_FRAME
        )

        # Build per-side Command pytree class from configured ActionComponents.
        # Component instances are constructed with scenario-derived knobs
        # (truth_dynamics, frames, mass tracking) so the env step's fold can
        # call component.apply(...) without re-resolving config every tick.
        def _build_component_instances(component_keys, track_mass: bool):
            instances = []
            classes = []
            for key in component_keys:
                cls = _ACTION_COMP_LOOKUP[key]
                if cls is ImpulsiveManeuver:
                    inst = ImpulsiveManeuver(
                        action_frame=self.action_frame,
                        truth_frame=self.truth_frame,
                        track_mass=track_mass,
                    )
                elif cls is Communicate:
                    inst = Communicate()
                elif cls is AttitudeControl:
                    rotation_dim = 1 if self.action_frame is Frame.RT else 3
                    inst = AttitudeControl(
                        rotation_dim=rotation_dim,
                        torque_max=self.config.attitude_control_torque_max,
                    )
                elif cls is PointAt:
                    inst = PointAt(boresight_body=self.config.pointing_boresight_body)
                else:
                    raise ValueError(f"No instance constructor for action component {key!r}")
                instances.append(inst)
                classes.append(cls)
            return tuple(instances), tuple(classes)

        guard_instances, _ = _build_component_instances(
            config.guard_action_components, guard_track_mass
        )
        bandit_instances, _ = _build_component_instances(
            config.bandit_action_components, bandit_track_mass
        )
        self.guard_action_component_instances = guard_instances
        self.bandit_action_component_instances = bandit_instances
        # Pass instances (not classes) so frame-aware ImpulsiveManeuver can
        # report the correct dv shape (2 for RT, 3 for RTN/ECI).
        self.guard_command_cls = build_command_class(
            guard_instances, config.n_guards, "GuardCommand"
        )
        self.bandit_command_cls = build_command_class(
            bandit_instances, config.n_bandits, "BanditCommand"
        )

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

        # Preflight: components depending on COMMUNICATE wiring (CommsLeak
        # observation reads guard `active`; LbgWithCommsReward reads it for
        # the comm-cost term) must have COMMUNICATE in guard_action_components.
        # Catching this at construction beats a late AttributeError downstream.
        self._validate_comms_wiring()

    def _validate_comms_wiring(self) -> None:
        """Fail fast when CommsLeak / LbgWithComms is wired without COMMUNICATE.

        Both consumers read ``actions.sides.guard.active``; without the
        Communicate component on the guard side, the field doesn't exist on
        the assembled Command pytree and the call would raise AttributeError
        later, often deep inside a jit trace where the message is opaque.
        """
        from orbitalgym.observations.comms_leak import CommsLeakObservation
        from orbitalgym.observations.composite import CompositeObservation
        from orbitalgym.rewards.lbg_with_comms import LbgWithCommsReward

        def _walk_observation(fn) -> bool:
            if isinstance(fn, CommsLeakObservation):
                return True
            if isinstance(fn, CompositeObservation):
                return any(_walk_observation(c) for c in fn.constituents)
            return False

        needs_comms = (
            _walk_observation(self.guard_observation_fn)
            or _walk_observation(self.bandit_observation_fn)
            or isinstance(self.reward_fn, LbgWithCommsReward)
        )
        if not needs_comms:
            return
        if ActionComponentKey.COMMUNICATE not in self.config.guard_action_components:
            raise ValueError(
                "CommsLeakObservation / LbgWithCommsReward requires "
                "ActionComponentKey.COMMUNICATE in guard_action_components "
                "(both consumers read action.sides.guard.active). Got: "
                f"guard_action_components={self.config.guard_action_components!r}"
            )

    def reset(self, key: jax.Array) -> tuple[EnvState, BySide]:
        """Reset returns (env_state, BySide(guard=SideOutput, bandit=SideOutput))."""
        k_ic, k_obs_g, k_obs_b = jax.random.split(key, 3)
        guards, bandits, ok = self._reset_with_icspec(k_ic)
        # Re-materialize derived views from the canonical truth-frame field so
        # observations and beliefs at t=0 see fields that are mutually
        # consistent. The sampler may also have populated them, but we make the
        # truth field authoritative.
        ref_eci6 = jnp.concatenate(
            [
                self.config.reference_orbit.position_eci,
                self.config.reference_orbit.velocity_eci,
            ]
        )
        truth_field_name = _truth_field(self.truth_frame)
        guards = _materialize_derived_views(
            guards,
            truth_field_name,
            self.truth_frame,
            self.guard_extended_frames,
            ref_eci6,
        )
        bandits = _materialize_derived_views(
            bandits,
            truth_field_name,
            self.truth_frame,
            self.bandit_extended_frames,
            ref_eci6,
        )
        state = EnvState(
            t=jnp.asarray(0.0),
            step=jnp.asarray(0),
            guards=guards,
            bandits=bandits,
            reference_orbit=self.config.reference_orbit,
            ic_valid=ok,
        )
        identity_actions = Actions(
            sides=BySide(
                guard=self.guard_command_cls.zeros(self.config.n_guards),
                bandit=self.bandit_command_cls.zeros(self.config.n_bandits),
            )
        )
        obs_g = self.guard_observation_fn(
            state, identity_actions, Side.GUARD, self.config, k_obs_g, state.t
        )
        obs_b = self.bandit_observation_fn(
            state, identity_actions, Side.BANDIT, self.config, k_obs_b, state.t
        )
        initial_done = jnp.asarray(False)
        initial_outputs = BySide(
            guard=SideOutput(obs=obs_g, reward=jnp.asarray(0.0), done=initial_done),
            bandit=SideOutput(obs=obs_b, reward=jnp.asarray(0.0), done=initial_done),
        )
        return state, initial_outputs

    def reset_from_state(self, state: EnvState, key: jax.Array) -> tuple[EnvState, BySide]:
        """Start an episode from a stored state instead of sampling one.

        Returns the state with the clock zeroed, derived frame views
        re-materialized from the truth frame, and the initial observations
        computed with ``key``. The reference orbit carried by ``state`` is
        kept so a bank sampled under one epoch replays under the same one.
        """
        k_obs_g, k_obs_b = jax.random.split(key, 2)
        ref_eci6 = jnp.concatenate(
            [state.reference_orbit.position_eci, state.reference_orbit.velocity_eci]
        )
        truth_field_name = _truth_field(self.truth_frame)
        guards = _materialize_derived_views(
            state.guards, truth_field_name, self.truth_frame, self.guard_extended_frames, ref_eci6
        )
        bandits = _materialize_derived_views(
            state.bandits, truth_field_name, self.truth_frame, self.bandit_extended_frames, ref_eci6
        )
        restored = state.replace(  # pyrefly: ignore[missing-attribute]
            t=jnp.zeros_like(state.t),
            step=jnp.zeros_like(state.step),
            guards=guards,
            bandits=bandits,
        )
        identity_actions = Actions(
            sides=BySide(
                guard=self.guard_command_cls.zeros(self.config.n_guards),
                bandit=self.bandit_command_cls.zeros(self.config.n_bandits),
            )
        )
        obs_g = self.guard_observation_fn(
            restored, identity_actions, Side.GUARD, self.config, k_obs_g, restored.t
        )
        obs_b = self.bandit_observation_fn(
            restored, identity_actions, Side.BANDIT, self.config, k_obs_b, restored.t
        )
        initial_done = jnp.asarray(False)
        outputs = BySide(
            guard=SideOutput(obs=obs_g, reward=jnp.asarray(0.0), done=initial_done),
            bandit=SideOutput(obs=obs_b, reward=jnp.asarray(0.0), done=initial_done),
        )
        return restored, outputs

    def _reset_with_icspec(self, k_ic: jax.Array):
        """Rejection-sampling reset for ICSpec configurations.

        Runs `jax.lax.while_loop` capped at `ic_sampler.max_attempts`. If all
        validators pass, returns (guards, bandits, True). If the cap is exhausted,
        returns the last-drawn (guards, bandits) with ic_valid=False — non-fatal,
        so downstream code can surface infeasibility rates instead of crashing.
        """
        n_guards = self.config.n_guards
        n_bandits = self.config.n_bandits
        # Use the auto-extended tuples so the sampler builds the same per-side
        # state class the env uses; otherwise pytree structures won't match.
        guard_components = self.config.guard_components_extended
        bandit_components = self.config.bandit_components_extended

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

        # Pre-propagation reference, used to rotate the impulsive Δv (which is
        # applied at the START of the step). The post-propagation reference is
        # only canonical for the post-step state and its derived views.
        ref_pre_eci6 = jnp.concatenate(
            [
                state.reference_orbit.position_eci,
                state.reference_orbit.velocity_eci,
            ]
        )

        # Propagate the reference orbit one tick. Reference is always 1 vehicle in ECI.
        ref_state6 = ref_pre_eci6[None, :]  # (1, 6)
        ref_dv = jnp.zeros((1, 3))
        ref_next6 = self.reference_orbit_dynamics(
            ref_state6, ref_dv, params=None, dt=self.config.dt
        )
        ref_next = state.reference_orbit.replace(  # pyrefly: ignore[missing-attribute]
            position_eci=ref_next6[0, :3],
            velocity_eci=ref_next6[0, 3:],
        )

        # Per-side action-component fold. Each component reads its slice of
        # the side's Command pytree and returns the post-component side state.
        # ImpulsiveManeuver writes applied_dv (Δv in truth-frame, padded to width 3).
        # The explicit translational dynamics block below then propagates both sides.
        next_guards = state.guards
        for component in self.guard_action_component_instances:
            next_guards = component.apply(
                actions.sides.guard,
                next_guards,
                self.guard_params,
                self.config.dt,
                ref_pre_eci6,
                k_dyn,
            )

        next_bandits = state.bandits
        for component in self.bandit_action_component_instances:
            next_bandits = component.apply(
                actions.sides.bandit,
                next_bandits,
                self.bandit_params,
                self.config.dt,
                ref_pre_eci6,
                k_dyn,
            )

        # Translational dynamics — always runs. Action-free sides (no
        # ImpulsiveManeuver) have applied_dv=0, giving pure free-drift. Sides
        # with an ImpulsiveManeuver had their applied_dv written by the fold
        # above. After propagation, applied_dv is zeroed so it does not
        # accumulate across steps.
        truth_field_name = _truth_field(self.truth_frame)

        def _propagate_translation(side_state, params):
            truth_arr = getattr(side_state, truth_field_name)
            applied_dv = side_state.applied_dv
            # Slice to truth-frame width (2 for RT, 3 for RTN/ECI).
            dv = applied_dv[:, : self.truth_frame.dim]
            new_truth = self.truth_dynamics(truth_arr, dv, params, self.config.dt)
            return side_state.replace(
                **{truth_field_name: new_truth},
                applied_dv=jnp.zeros_like(applied_dv),
            )

        next_guards = _propagate_translation(next_guards, self.guard_params)
        next_bandits = _propagate_translation(next_bandits, self.bandit_params)

        # Attitude dynamics — runs iff configured. Free precession is the zero-torque path;
        # AttitudeControl actions populate applied_torque earlier in the step.
        # Each side is guarded independently so asymmetric configurations (e.g. only
        # the guard has ATTITUDE+BODY_RATES while the bandit has only RTN) do not
        # raise AttributeError when the other side lacks .quat/.omega/.applied_torque.
        if self.attitude_dynamics is not None:
            if self.guard_has_attitude:
                quat_g, omega_g = self.attitude_dynamics(
                    next_guards.quat,
                    next_guards.omega,
                    next_guards.applied_torque,
                    self.guard_attitude_params,
                    self.config.dt,
                )
                next_guards = next_guards.replace(
                    quat=quat_g,
                    omega=omega_g,
                    applied_torque=jnp.zeros_like(next_guards.applied_torque),
                )

            if self.bandit_has_attitude:
                quat_b, omega_b = self.attitude_dynamics(
                    next_bandits.quat,
                    next_bandits.omega,
                    next_bandits.applied_torque,
                    self.bandit_attitude_params,
                    self.config.dt,
                )
                next_bandits = next_bandits.replace(
                    quat=quat_b,
                    omega=omega_b,
                    applied_torque=jnp.zeros_like(next_bandits.applied_torque),
                )

        # Materialize every non-truth frame from the post-step truth array. Use
        # the *propagated* reference orbit (ref_next) so derived views are
        # consistent with the truth state at t+dt. HCW-only scenarios skip this
        # because the only spatial frame is the truth frame.
        ref_eci6 = jnp.concatenate([ref_next.position_eci, ref_next.velocity_eci])
        next_guards = _materialize_derived_views(
            next_guards,
            truth_field_name,
            self.truth_frame,
            self.guard_extended_frames,
            ref_eci6,
        )
        next_bandits = _materialize_derived_views(
            next_bandits,
            truth_field_name,
            self.truth_frame,
            self.bandit_extended_frames,
            ref_eci6,
        )

        next_state = state.replace(  # pyrefly: ignore[missing-attribute]
            t=state.t + self.config.dt,
            step=state.step + 1,
            guards=next_guards,
            bandits=next_bandits,
            reference_orbit=ref_next,
        )

        # Per-side observations + rewards.
        obs_g = self.guard_observation_fn(
            next_state, actions, Side.GUARD, self.config, k_obs_g, next_state.t
        )
        obs_b = self.bandit_observation_fn(
            next_state, actions, Side.BANDIT, self.config, k_obs_b, next_state.t
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
