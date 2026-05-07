"""Composable state components.

Each component contributes a set of named fields to a vehicle's per-scenario state
class (built by state.assemble.build_state_class). Components are orthogonal —
a scenario may combine RTNState + Mass + Power, or just RTState, etc.

Attitude and BodyRates ship as protocol-conforming stubs (no dynamics integration
in the bootstrap); they exist so future work can drop in without changing the
assembly machinery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class StateComponent(Protocol):
    """Shape contract for a state component."""

    name: ClassVar[str]

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        """Per-vehicle shape of each contributed field."""
        ...

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        """Zero-initialize all fields for `n` vehicles. Returns dict field -> (n, *field_shape)."""
        ...


class RTState:
    """2D in-plane relative state in the RT (radial + along-track) sub-frame.

    rt layout: [r, t, r_dot, t_dot] — 4-dim per vehicle.
    """

    name: ClassVar[str] = "rt"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"rt": (4,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"rt": jnp.zeros((n, 4))}


class RTNState:
    """Full 3D relative state in the RTN frame.

    rtn layout: [r, t, n, r_dot, t_dot, n_dot] — 6-dim per vehicle.
    """

    name: ClassVar[str] = "rtn"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"rtn": (6,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"rtn": jnp.zeros((n, 6))}


class ECIState:
    """Absolute 6D inertial state (Earth-Centered Inertial).

    eci layout: [rx, ry, rz, vx, vy, vz] — position [m] then velocity [m/s].
    """

    name: ClassVar[str] = "eci"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"eci": (6,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"eci": jnp.zeros((n, 6))}


class Mass:
    """Propellant-mass tracking.

    Only propellant_mass is state; wet_mass is derived on demand as
    params.dry_mass_kg + state.propellant_mass.
    """

    name: ClassVar[str] = "mass"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"propellant_mass": ()}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"propellant_mass": jnp.zeros((n,))}


class Power:
    """Scalar state-of-charge. No-op dynamics in bootstrap (stub)."""

    name: ClassVar[str] = "power"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"charge": ()}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"charge": jnp.zeros((n,))}


class Attitude:
    """Quaternion attitude (w, x, y, z). Protocol-level stub in bootstrap."""

    name: ClassVar[str] = "attitude"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"quat": (4,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        # Initialize to identity quaternion (1, 0, 0, 0).
        z = jnp.zeros((n, 4))
        return {"quat": z.at[:, 0].set(1.0)}


class BodyRates:
    """Body-frame angular rates (ωx, ωy, ωz). Protocol-level stub in bootstrap."""

    name: ClassVar[str] = "body_rates"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"omega": (3,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"omega": jnp.zeros((n, 3))}


class AppliedDV:
    """Per-step applied translational Δv (m/s) in the truth frame.

    Transient: written by ImpulsiveManeuver.apply (or any other action
    component that produces Δv), consumed by the translational dynamics
    block in env.step, then zeroed.

    Width is always 3 on state for shape uniformity. The dynamics step
    slices [:, :2] for Frame.RT, [:, :3] for RTN/ECI.
    """

    name: ClassVar[str] = "applied_dv"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"applied_dv": (3,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"applied_dv": jnp.zeros((n, 3))}


class AppliedTorque:
    """Per-step applied body-frame torque (N·m).

    Transient: written by AttitudeControl.apply (or any other torque-
    producing action component), consumed by rigid_body_attitude_step in
    env.step, then zeroed.
    """

    name: ClassVar[str] = "applied_torque"

    @staticmethod
    def fields() -> Mapping[str, tuple[int, ...]]:
        return {"applied_torque": (3,)}

    @staticmethod
    def zeros(n: int) -> Mapping[str, jax.Array]:
        return {"applied_torque": jnp.zeros((n, 3))}
