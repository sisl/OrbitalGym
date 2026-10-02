"""Linear feedback baselines on the HCW relative state: PD and LQR.

Both laws regulate the six-dimensional RTN state ``x = [r; v]`` of a vehicle
relative to a target, the nearest opposing vehicle or the origin of the frame,
toward zero. They command one impulse per step, applied at the start of the
step, so the relative state obeys ``x_next = A(dt) (x + B u)`` with
``B = [0; I]`` while the target coasts.

The PD law applies proportional and derivative feedback on the relative
position and velocity,

    u = -dt (kp r + kd v)

with gains in acceleration form. The defaults ``kp = 1e-4 s^-2`` and
``kd = 0.02 s^-1`` give, for a continuous-time double integrator without
saturation, a damping ratio of one and a time constant of 100 s.

The LQR law applies ``u = -K x``, where ``K`` solves the discrete-time
infinite-horizon regulator problem for the transition above with state weight
``diag(1 / position_scale_m^2, 1 / velocity_scale_mps^2)`` and impulse weight
``1 / impulse_scale_mps^2``. The defaults scale the position by 100 m, the
velocity by 1 m/s, and the impulse by 0.1 m/s. The regulator has no integral
action, so a target under constant thrust leaves a steady-state offset.

Neither law accounts for the impulse limit; the policies scale the commanded
impulse down to ``max_dv_mps`` along its direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are

from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.policies.heuristic.glideslope import clip_to_cap
from orbitalgym.policies.heuristic.views import mean_from_view
from orbitalgym.registry import PolicyKey, register

PD_KP = 1e-4
PD_KD = 0.02


def pd_impulse(relative: jax.Array, dt: float, kp: float = PD_KP, kd: float = PD_KD) -> jax.Array:
    """PD impulse for relative states ``[r; v]`` of shape ``(..., 6)``; not limited."""
    return -dt * (kp * relative[..., :3] + kd * relative[..., 3:])


def hcw_lqr_gain(
    mean_motion: float,
    dt: float,
    position_scale_m: float = 100.0,
    velocity_scale_mps: float = 1.0,
    impulse_scale_mps: float = 0.1,
) -> jax.Array:
    """The ``(3, 6)`` regulator gain for the impulsive HCW transition over one step ``dt``."""
    a = np.asarray(hcw_rtn_stm(mean_motion, dt))
    b = a[:, 3:]
    q = np.diag([1 / position_scale_m**2] * 3 + [1 / velocity_scale_mps**2] * 3)
    r = np.eye(3) / impulse_scale_mps**2
    p = solve_discrete_are(a, b, q, r)
    return jnp.asarray(np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a))


def lqr_impulse(gain: jax.Array, relative: jax.Array) -> jax.Array:
    """LQR impulse ``-K x`` for relative states of shape ``(..., 6)``; not limited."""
    return -jnp.einsum("ij,...j->...i", gain, relative)


def _relative_to_target(
    agent_view: Any, n_vehicles: int, n_opponents: int, to_origin: bool
) -> jax.Array:
    """Each vehicle's RTN state minus its target's: the origin or the nearest opponent."""
    mean = mean_from_view(agent_view, n_vehicles, n_opponents, 6)
    idx = jnp.arange(n_vehicles)
    own = mean[idx, idx, :]
    if to_origin:
        return own
    opp = mean[:, n_vehicles:, :]
    nearest = jnp.argmin(jnp.linalg.norm(own[:, None, :3] - opp[..., :3], axis=-1), axis=-1)
    return own - opp[idx, nearest, :]


def _wrap(dv: jax.Array, command_cls: Any, n_vehicles: int):
    """Fill a zero command with the impulse, cut to the command's ``dv`` width."""
    template = command_cls.zeros(n_vehicles)
    return template.replace(dv=dv[:, : template.dv.shape[-1]].astype(template.dv.dtype))


@register(PolicyKey.PD_FEEDBACK)
@dataclass(frozen=True)
class PDFeedback:
    """PD feedback onto the nearest opposing vehicle, or onto the origin.

    Each vehicle regulates its RTN state relative to its target with
    :func:`pd_impulse` and limits the impulse to ``max_dv_mps``. The target is
    the nearest opposing vehicle, or the origin of the frame (the lady in
    lady-bandit-guard) when ``to_origin`` is set.

    The agent view is either a belief with ``mean (N_obs, N_total, 6)`` or a
    flat full-state observation of the same numbers, read as
    :class:`~orbitalgym.policies.heuristic.glideslope.GlideslopeToLady` reads it.
    """

    dt: float
    max_dv_mps: float
    kp: float
    kd: float
    to_origin: bool
    n_vehicles: int
    n_opponents: int
    command_cls: Any

    @classmethod
    def build(
        cls,
        *,
        dt: float,
        n_vehicles: int,
        n_opponents: int,
        command_cls: Any,
        max_dv_mps: float,
        kp: float = PD_KP,
        kd: float = PD_KD,
        to_origin: bool = False,
    ) -> PDFeedback:
        return cls(
            dt=float(dt),
            max_dv_mps=float(max_dv_mps),
            kp=float(kp),
            kd=float(kd),
            to_origin=bool(to_origin),
            n_vehicles=n_vehicles,
            n_opponents=n_opponents,
            command_cls=command_cls,
        )

    def __call__(self, policy_state: Any, agent_view: Any, key: jax.Array, t: jax.Array):
        del key, t
        relative = _relative_to_target(
            agent_view, self.n_vehicles, self.n_opponents, self.to_origin
        )
        dv = clip_to_cap(pd_impulse(relative, self.dt, self.kp, self.kd), self.max_dv_mps)
        return _wrap(dv, self.command_cls, self.n_vehicles), policy_state


@register(PolicyKey.LQR_FEEDBACK)
@dataclass(frozen=True)
class LQRFeedback:
    """LQR feedback onto the nearest opposing vehicle, or onto the origin.

    Each vehicle regulates its RTN state relative to its target with the gain
    of :func:`hcw_lqr_gain` and limits the impulse to ``max_dv_mps``. The
    target and the agent view are those of :class:`PDFeedback`.

    ``gain`` holds the ``(3, 6)`` regulator gain as nested tuples so that the
    policy stays hashable; :meth:`build` computes it from the mean motion, the
    step, and the weight scales.
    """

    gain: tuple[tuple[float, ...], ...]
    max_dv_mps: float
    to_origin: bool
    n_vehicles: int
    n_opponents: int
    command_cls: Any

    @classmethod
    def build(
        cls,
        *,
        mean_motion: float,
        dt: float,
        n_vehicles: int,
        n_opponents: int,
        command_cls: Any,
        max_dv_mps: float,
        position_scale_m: float = 100.0,
        velocity_scale_mps: float = 1.0,
        impulse_scale_mps: float = 0.1,
        to_origin: bool = False,
    ) -> LQRFeedback:
        gain = hcw_lqr_gain(
            mean_motion, dt, position_scale_m, velocity_scale_mps, impulse_scale_mps
        )
        return cls(
            gain=tuple(tuple(float(v) for v in row) for row in np.asarray(gain)),
            max_dv_mps=float(max_dv_mps),
            to_origin=bool(to_origin),
            n_vehicles=n_vehicles,
            n_opponents=n_opponents,
            command_cls=command_cls,
        )

    def __call__(self, policy_state: Any, agent_view: Any, key: jax.Array, t: jax.Array):
        del key, t
        relative = _relative_to_target(
            agent_view, self.n_vehicles, self.n_opponents, self.to_origin
        )
        dv = clip_to_cap(lqr_impulse(jnp.asarray(self.gain), relative), self.max_dv_mps)
        return _wrap(dv, self.command_cls, self.n_vehicles), policy_state
