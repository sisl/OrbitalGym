"""Keplerian (point-mass two-body) orbit dynamics in ECI.

`KeplerianEciDynamics` is a typed-instance dataclass: the reference epoch is
a per-instance field (defaulting to J2000), so different scenarios can use
different epochs without sharing module state. The Epoch object and the
astrojax dynamics RHS are built once in ``__post_init__`` (Python time, before
any JIT trace) — this avoids both module-level state and the
``ConcretizationTypeError`` that would result from calling ``Epoch(...)``
inside a JIT-traced step (its internal calendar→MJD conversion uses Python
conditionals over int args).

Registered against ``DynamicsKey.KEPLERIAN_ECI`` so callers can pass either
the typed instance directly (recommended — lets you bind the scenario's
epoch) or the bare ``DynamicsKey`` (which the env resolves to a default-
constructed J2000 instance, see ``env.core._resolve_dynamics_callable``).

Δv is applied at the START of the interval (impulsive convention shared with
HCW), so impulsive and continuous-actuator step functions remain
interchangeable under a common signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import astrojax
import jax
from astrojax import Epoch
from astrojax.eop import zero_eop

from orbitalgym.registry import DynamicsKey, DynamicsKind, Frame, register


@register(DynamicsKey.KEPLERIAN_ECI, frame=Frame.ECI, kind=DynamicsKind.ABSOLUTE)
@dataclass(frozen=True)
class KeplerianEciDynamics:
    """Point-mass two-body Keplerian dynamics in ECI, parameterized by epoch.

    The Epoch and RHS are built in ``__post_init__`` (Python time) and stored
    on the instance. Calls into ``__call__`` then propagate without
    constructing astrojax objects inside JIT, which would otherwise fail with
    ``ConcretizationTypeError`` from internal calendar→MJD conversions.

    For point-mass two-body, ``epoch_mjd_utc`` is physics-irrelevant (the RHS
    has no time dependence), but it remains a per-instance knob so that
    ``ScenarioConfig.__post_init__`` can wire ``cfg.epoch_mjd_utc`` through
    consistently for all dynamics kinds.

    Attributes:
        epoch_mjd_utc: Reference epoch in MJD UTC. Default 51544.5 (J2000 noon).
        sub_steps: Fixed sub-steps per env tick. ``dt`` is divided evenly across them.

    Class-level (not dataclass-field) attributes ``frame`` / ``kind`` satisfy
    the registry validator that other dynamics (registered functions) get
    via the ``@register(...)`` decorator.
    """

    epoch_mjd_utc: float = 51544.5
    sub_steps: int = 1
    frame: ClassVar[Frame] = Frame.ECI
    kind: ClassVar[DynamicsKind] = DynamicsKind.ABSOLUTE
    # Excluded-from-eq/hash: derived RHS object. Set by __post_init__.
    _rhs: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self.sub_steps < 1:
            raise ValueError(f"sub_steps must be >= 1; got {self.sub_steps}")
        # MJD-to-calendar conversion for the Epoch ctor would require care; for
        # point-mass two-body the epoch is irrelevant, so we anchor at J2000.
        # Future: thread epoch_mjd_utc into harmonic-gravity scenarios.
        epoch = Epoch(2000, 1, 1, 12, 0, 0.0)
        rhs = astrojax.create_orbit_dynamics(eop=zero_eop(), epoch_0=epoch)
        # Frozen dataclass: bypass setattr.
        object.__setattr__(self, "_rhs", rhs)

    def _step_one(self, state6: jax.Array, dt: float) -> jax.Array:
        sub_dt = dt / self.sub_steps
        s = state6
        for _ in range(self.sub_steps):
            s = astrojax.rk4_step(self._rhs, 0.0, s, sub_dt).state
        return s

    def __call__(self, state: jax.Array, dv: jax.Array, params: Any, dt: float) -> jax.Array:
        """Keplerian (point-mass) ECI step.

        state: (n, 6) — (rx, ry, rz, vx, vy, vz) per vehicle in ECI [m, m/s].
        dv:    (n, 3) — (dvx, dvy, dvz) impulsive Δv applied at start of interval.
        params: unused (kept for signature parity with other dynamics).
        dt:    scalar seconds.
        """
        del params
        s0 = state.at[:, 3:].add(dv)
        return jax.vmap(self._step_one, in_axes=(0, None))(s0, dt)
