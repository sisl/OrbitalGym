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

from orbital_game.registry import DynamicsKey, DynamicsKind, Frame, register

# Reference epoch for the dynamics closure. Point-mass two-body has no explicit
# time dependence; spherical-harmonics gravity rotates with Earth (so epoch
# matters for ECEF/ECI resolution), but for the purposes of this wrapper we
# anchor at J2000 — callers needing custom epoch handling should use astrojax
# directly. This mirrors the choice in ``keplerian.py``.
_EPOCH_REF = Epoch(2000, 1, 1, 12, 0, 0.0)

_INTEGRATORS = {
    "rk4": astrojax.rk4_step,
    "dp54": astrojax.dp54_step,
    "rkf45": astrojax.rkf45_step,
}

# Module-level cache keyed by ``ForceModelConfig`` so that re-instantiating
# ``AstrojaxOrbitDynamics`` with the same force model does not rebuild the
# (potentially expensive) spherical-harmonics closure.
#
# Thread safety: keyed by ``ForceModelConfig`` identity/equality. JAX users in
# this project are single-threaded; under the GIL the worst case is a redundant
# ``create_orbit_dynamics`` call when two threads race the cache miss. Both
# threads get a functionally identical RHS; harmless. No lock needed.
_RHS_CACHE: dict[ForceModelConfig, Any] = {}


def _get_rhs(force_model: ForceModelConfig):
    rhs = _RHS_CACHE.get(force_model)
    if rhs is None:
        rhs = astrojax.create_orbit_dynamics(
            eop=zero_eop(),
            epoch_0=_EPOCH_REF,
            config=force_model,
        )
        _RHS_CACHE[force_model] = rhs
    return rhs


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

    def _step_one(self, state6: jax.Array, dt: float) -> jax.Array:
        """Single fixed-step propagation on a 6D ECI state vector."""
        rhs = _get_rhs(self.force_model)
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
