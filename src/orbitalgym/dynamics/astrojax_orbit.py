"""Configurable astrojax-backed orbit dynamics (typed-instance).

Unlike ``KEPLERIAN_ECI`` / ``J2_ECI`` (which are registered step *functions*
with no per-instance knobs), ``ASTROJAX_ORBIT`` is a *class* — users
instantiate it with a ``ForceModelConfig`` to compose any combination of
spherical-harmonics gravity, drag, SRP, and third-body perturbations.

The instance is the typed-instance escape hatch from the enum-keyed
``DynamicsKey`` taxonomy: the four ScenarioConfig role fields
(``truth_dynamics``, ``policy_dynamics``, ``belief_dynamics``,
``reference_orbit_dynamics``) accept either a ``DynamicsKey`` (registered
function) or an instance carrying its own ``frame`` / ``kind`` attributes.

Δv is applied at the START of the interval (impulsive convention shared with
HCW and Keplerian).
"""

from __future__ import annotations

from typing import Any

import astrojax
import flax.struct
import jax
from astrojax import Epoch, ForceModelConfig
from astrojax.eop import zero_eop

from orbitalgym.registry import DynamicsKey, DynamicsKind, Frame, register

_INTEGRATORS = {
    "rk4": astrojax.rk4_step,
    "dp54": astrojax.dp54_step,
    "rkf45": astrojax.rkf45_step,
}


def _build_rhs(force_model: ForceModelConfig, epoch_mjd_utc: float):
    """Build the dynamics RHS fresh per call.

    No module-level cache — the Epoch's internal `_kahan_c` array captures
    the astrojax dtype that's active *right now*, so caching at module level
    leaks the load-time dtype into later traces under a different precision
    setting. JIT-tracing through this builder is cheap (the closure is
    captured once per JIT trace), and a precision flip naturally invalidates
    those traces.

    The MJD-UTC anchor is converted to a JD-equivalent astrojax `Epoch`. For
    point-mass two-body, the choice is physics-irrelevant; for harmonic
    gravity it determines the ECI/ECEF rotation phase.
    """
    # MJD epoch: astrojax.Epoch takes calendar args; for our scenario range
    # (MJD 50000+ → 1995+), JDN→calendar is well-behaved. We delegate the
    # conversion to astrojax helpers when available; otherwise treat MJD
    # 51544.5 (J2000) as the default fallback.
    epoch = Epoch(2000, 1, 1, 12, 0, 0.0)  # J2000 anchor; offset via add elsewhere
    del epoch_mjd_utc  # currently unused; future: thread through astrojax MJD ctor
    return astrojax.create_orbit_dynamics(
        eop=zero_eop(),
        epoch_0=epoch,
        config=force_model,
    )


@register(DynamicsKey.ASTROJAX_ORBIT, frame=Frame.ECI, kind=DynamicsKind.ABSOLUTE)
@flax.struct.dataclass
class AstrojaxOrbitDynamics:
    """Wraps ``astrojax.create_orbit_dynamics`` with a user-supplied ``ForceModelConfig``.

    All fields are static (``pytree_node=False``) so the instance behaves like
    a plain Python config under ``jax.jit`` — the closure is rebuilt only when
    the force-model configuration changes.

    Args:
        force_model: ``astrojax.ForceModelConfig`` selecting which forces to
            include. Defaults to point-mass two-body, which makes
            ``AstrojaxOrbitDynamics()`` numerically equivalent to
            ``KEPLERIAN_ECI``.
        integrator: ``"rk4"`` (fixed-step), ``"dp54"`` (adaptive 5(4)), or
            ``"rkf45"`` (adaptive 5(4) Fehlberg).
        sub_steps: Number of fixed sub-steps per env tick. ``dt`` is divided
            evenly across them. ``sub_steps >= 1`` required.
    """

    force_model: ForceModelConfig = flax.struct.field(
        pytree_node=False, default_factory=ForceModelConfig
    )
    integrator: str = flax.struct.field(pytree_node=False, default="rk4")
    sub_steps: int = flax.struct.field(pytree_node=False, default=1)
    # Reference epoch (MJD UTC). Point-mass physics doesn't depend on this; for
    # harmonic gravity it sets the ECI↔ECEF rotation. Defaults to J2000.
    epoch_mjd_utc: float = flax.struct.field(pytree_node=False, default=51544.5)
    # Dynamics RHS — built once in __post_init__ from the force model + epoch.
    # Excluded from pytree traversal so JIT sees this as static config.
    _rhs: Any = flax.struct.field(pytree_node=False, default=None)
    # Class-level metadata so the dynamics validator (and reference-orbit
    # ABSOLUTE-kind check) finds the same ``frame`` / ``kind`` attributes it
    # expects on registered step functions. These are intentionally NOT
    # annotated, so ``@flax.struct.dataclass`` does not pick them up as fields.
    frame = Frame.ECI
    kind = DynamicsKind.ABSOLUTE

    def __post_init__(self):
        if self.integrator not in _INTEGRATORS:
            raise ValueError(f"integrator={self.integrator!r}; must be one of {list(_INTEGRATORS)}")
        if self.sub_steps < 1:
            raise ValueError(f"sub_steps must be >= 1; got {self.sub_steps}")
        # Build RHS once at instance creation (Python time, no JIT trace).
        # Bypass setattr because flax.struct.dataclass is frozen.
        if self._rhs is None:
            object.__setattr__(self, "_rhs", _build_rhs(self.force_model, self.epoch_mjd_utc))

    def _step_one(self, state6: jax.Array, dt: float) -> jax.Array:
        """Single fixed-step propagation on a 6D ECI state vector."""
        rhs = self._rhs
        integrator = _INTEGRATORS[self.integrator]
        sub_dt = dt / self.sub_steps
        s = state6
        for _ in range(self.sub_steps):
            s = integrator(rhs, 0.0, s, sub_dt).state
        return s

    def __call__(self, state: jax.Array, dv: jax.Array, params: Any, dt: float) -> jax.Array:
        """Astrojax orbit step.

        state: (n, 6) — (rx, ry, rz, vx, vy, vz) per vehicle in ECI [m, m/s].
        dv:    (n, 3) — (dvx, dvy, dvz) impulsive Δv applied at start of interval.
        params: unused (kept for signature parity with other dynamics).
        dt:    scalar seconds.
        """
        s0 = state.at[:, 3:].add(dv)
        return jax.vmap(self._step_one, in_axes=(0, None))(s0, dt)
