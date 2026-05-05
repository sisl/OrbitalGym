"""ActionComponent protocol + concrete components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from orbital_game.frames.conversions import convert_action
from orbital_game.registry import Frame

G0 = 9.80665  # m/s^2 — standard gravity for Isp calcs


@runtime_checkable
class ActionComponent(Protocol):
    """Composable per-side action-pytree component.

    Mirrors `StateComponent` for the schema half (`fields`, `zeros`). Adds
    `apply(command, side_state, side_params, dt, ref_eci6, key) -> side_state`
    which is invoked by `OrbitalGameEnv.step` once per side, in component
    registration order. Components are responsible for any state mutation
    (including invoking dynamics — see `ImpulsiveManeuver`); the env loop only folds.
    """

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        """Per-agent shape of each contributed command field."""
        ...

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
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
        """Return the post-component side_state. May invoke dynamics."""
        ...


@dataclass(frozen=True)
class ImpulsiveManeuver:
    """Impulsive Δv applied at the start of the step + truth-dynamics propagation.

    Applies an impulsive Δv (with optional propellant tracking via the rocket
    equation) and then invokes `truth_dynamics` to propagate the side state
    through `dt`. After this component runs, the side's truth-frame state
    field has been updated and propellant has been deducted.

    Init knobs (env wires these from ScenarioConfig):
        truth_dynamics: the step function (signature: state6, dv, params, dt -> state6')
        action_frame:   frame the policy emits Δv in
        truth_frame:    frame `truth_dynamics` integrates in
        track_mass:     whether to deduct propellant via the rocket equation
    """

    truth_dynamics: Callable
    action_frame: Frame
    truth_frame: Frame
    track_mass: bool = True

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"dv": (3,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"dv": jnp.zeros((n, 3))}

    def apply(
        self,
        command: Any,
        side_state: Any,
        side_params: Any,
        dt: float,
        ref_eci6: jax.Array,
        key: jax.Array,
    ) -> Any:
        del key  # impulsive maneuver is deterministic
        dv_action = command.dv

        dv_truth = convert_action(dv_action, self.action_frame, self.truth_frame, ref_eci6)

        truth_field = self.truth_frame.value  # "rt" / "rtn" / "eci"
        truth_arr = getattr(side_state, truth_field)
        new_truth = self.truth_dynamics(truth_arr, dv_truth, side_params, dt)
        new_state = side_state.replace(**{truth_field: new_truth})

        if self.track_mass:
            dv_mag = jnp.linalg.norm(dv_action, axis=-1)
            wet_mass = side_params.dry_mass_kg + side_state.propellant_mass
            delta_propellant = wet_mass * (1.0 - jnp.exp(-dv_mag / (side_params.isp_s * G0)))
            new_state = new_state.replace(
                propellant_mass=jnp.maximum(new_state.propellant_mass - delta_propellant, 0.0)
            )
        return new_state


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

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"active": (), "payload": (6,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
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
