"""ActionComponent protocol + concrete components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from orbitalgym.frames.conversions import convert_action
from orbitalgym.registry import Frame

G0 = 9.80665  # m/s^2 — standard gravity for Isp calcs


@runtime_checkable
class ActionComponent(Protocol):
    """Composable per-side action-pytree component.

    Mirrors `StateComponent` for the schema half (`fields`, `zeros`). Adds
    `apply(command, side_state, side_params, dt, ref_eci6, key) -> side_state`
    which is invoked by `OrbitalGymEnv.step` once per side, in component
    registration order. Components are responsible for any state mutation
    (including invoking dynamics — see `ImpulsiveManeuver`); the env loop only folds.

    ``fields`` and ``zeros`` are instance methods so frame-aware components
    (e.g. ImpulsiveManeuver, where ``self.action_frame.dim`` decides the dv
    shape) can vary their schema. Components without per-instance state can
    simply ignore ``self``.
    """

    def fields(self) -> Mapping[str, tuple[int, ...]]:
        """Per-agent shape of each contributed command field."""
        ...

    def zeros(self, n: int) -> Mapping[str, jax.Array]:
        """Identity command — what 'do nothing for this component' looks like.
        Returns dict field -> (n, *field_shape)."""
        ...

    def apply(
        self,
        command: Any,
        side_state: Any,
        side_params: Any,
        dt: float,
        ref_eci6: jax.Array,
        key: jax.Array,
    ) -> Any:
        """Return the post-component side_state. Must not invoke any dynamics
        — translational and attitude propagation are both owned by env.step."""
        ...


@dataclass(frozen=True)
class ImpulsiveManeuver:
    """Impulsive Δv producer.

    Reads ``command.dv`` (in ``action_frame``), converts to ``truth_frame``,
    optionally deducts propellant via the rocket equation, and writes the
    result onto ``side_state.applied_dv`` (always width 3, zero-padded for RT).

    Does NOT integrate the dynamics — ``env.step`` owns the translational
    dynamics call (see the explicit propagation block in ``OrbitalGymEnv.step``).

    Per-step delta-v magnitude is limited to ``max_thrust_n * dt / m_wet`` and
    to the delta-v obtainable from remaining propellant. Direction is preserved
    when clipping. An empty tank produces no delta-v.

    The Δv field shape follows ``action_frame.dim`` — 2 for RT (radial,
    along-track) and 3 for RTN/ECI. ``fields()`` and ``zeros()`` are instance
    methods (not staticmethods) so the assembled Command pytree carries the
    correct shape per scenario; ``build_command_class`` is given component
    *instances* so it can read this.

    Init knobs (env wires these from ScenarioConfig):
        action_frame: frame the policy emits Δv in
        truth_frame:  frame the truth dynamics integrates in
        track_mass:   whether to deduct propellant via the rocket equation
    """

    action_frame: Frame
    truth_frame: Frame
    track_mass: bool = True

    def fields(self) -> Mapping[str, tuple[int, ...]]:
        return {"dv": (self.action_frame.dim,)}

    def zeros(self, n: int) -> Mapping[str, jax.Array]:
        return {"dv": jnp.zeros((n, self.action_frame.dim))}

    def apply(
        self,
        command: Any,
        side_state: Any,
        side_params: Any,
        dt: float,
        ref_eci6: jax.Array,
        key: jax.Array,
    ) -> Any:
        del key
        dv_action = command.dv  # (n, action_frame.dim)
        dv_mag = jnp.linalg.norm(dv_action, axis=-1)  # (n,)
        dtype = dv_action.dtype
        dry_mass = jnp.asarray(side_params.dry_mass_kg, dtype=dtype)

        if self.track_mass:
            wet_mass = dry_mass + side_state.propellant_mass
            dv_available = side_params.isp_s * G0 * jnp.log(wet_mass / dry_mass)
        else:
            wet_mass = jnp.broadcast_to(dry_mass, dv_mag.shape)
            dv_available = jnp.full_like(dv_mag, jnp.inf)

        dv_cap = side_params.max_thrust_n * dt / wet_mass
        dv_limit = jnp.minimum(dv_cap, dv_available)
        safe_mag = jnp.maximum(dv_mag, jnp.asarray(1e-12, dtype=dtype))
        scale = jnp.where(dv_mag > 0.0, jnp.minimum(1.0, dv_limit / safe_mag), 1.0)
        dv_action = dv_action * scale[:, None]
        dv_mag = dv_mag * scale

        dv_truth = convert_action(dv_action, self.action_frame, self.truth_frame, ref_eci6)
        pad_width = 3 - self.truth_frame.dim
        dv_padded = jnp.pad(dv_truth, ((0, 0), (0, pad_width)))

        replacements: dict[str, Any] = {"applied_dv": dv_padded}
        if self.track_mass:
            delta_propellant = wet_mass * (1.0 - jnp.exp(-dv_mag / (side_params.isp_s * G0)))
            replacements["propellant_mass"] = jnp.maximum(
                side_state.propellant_mass - delta_propellant, 0.0
            )
        return side_state.replace(**replacements)


@dataclass(frozen=True)
class Communicate:
    """Decentralized-communication action component.

    Per-agent fields:
        active: bool — whether this agent communicates this step
        payload: (D,) — broadcast payload (e.g. agent's belief mean)

    Apply is state-identity: communication's effects are visible to other
    agents via the observation pipeline (which receives the full Actions
    pytree under the Phase 3 protocol change). The reward function reads
    `active` to charge a per-broadcast cost.

    Payload shape is fixed at (6,) (RTN position+velocity) — JAX requires
    static field shapes for jit-stability. Scenarios with different payload
    dim must register a subclass with overridden fields()/zeros().
    """

    def fields(self) -> Mapping[str, tuple[int, ...]]:
        return {"active": (), "payload": (6,)}

    def zeros(self, n: int) -> Mapping[str, jax.Array]:
        return {
            "active": jnp.zeros((n,), dtype=jnp.bool_),
            "payload": jnp.zeros((n, 6)),
        }

    def apply(
        self,
        command: Any,
        side_state: Any,
        side_params: Any,
        dt: float,
        ref_eci6: jax.Array,
        key: jax.Array,
    ) -> Any:
        del command, side_params, dt, ref_eci6, key
        return side_state


@dataclass(frozen=True)
class AttitudeControl:
    """Torque-commanding attitude control.

    Reads command.torque (length rotation_dim — 1 in RT, 3 in RTN), lifts to a
    3-vector in 2D (zero x,y), optionally clips per-axis to ±torque_max, writes
    the result onto side_state.applied_torque.

    Does NOT integrate dynamics — env.step invokes rigid_body_attitude_step
    after the action fold. Mirror of the unified action↔dynamics pattern set
    by ImpulsiveManeuver: action produces, env consumes.
    """

    rotation_dim: int  # 1 (RT) or 3 (RTN)
    torque_max: tuple[float, ...] | None = None  # per-axis actuator clip (length 3)

    def fields(self) -> Mapping[str, tuple[int, ...]]:
        return {"torque": (self.rotation_dim,)}

    def zeros(self, n: int) -> Mapping[str, jax.Array]:
        return {"torque": jnp.zeros((n, self.rotation_dim), dtype=jnp.float32)}

    def apply(
        self,
        command: Any,
        side_state: Any,
        side_params: Any,
        dt: float,
        ref_eci6: jax.Array,
        key: jax.Array,
    ) -> Any:
        del side_params, dt, ref_eci6, key
        tau = command.torque  # (n, rotation_dim)

        if self.rotation_dim == 1:
            # 2D: rotation only about body z (N axis). Lift to (n, 3) with [0, 0, τz].
            n = tau.shape[0]
            tau_3d = jnp.concatenate([jnp.zeros((n, 2), dtype=tau.dtype), tau], axis=1)
        elif self.rotation_dim == 3:
            tau_3d = tau
        else:
            raise ValueError(
                f"AttitudeControl: rotation_dim must be 1 or 3, got {self.rotation_dim}"
            )

        if self.torque_max is not None:
            tmax = jnp.asarray(self.torque_max, dtype=tau_3d.dtype)
            tau_3d = jnp.clip(tau_3d, -tmax, tmax)

        return side_state.replace(applied_torque=tau_3d)
