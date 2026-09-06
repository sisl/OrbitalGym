"""Saturation-aware glideslope guidance toward the lady and onto the nearest opponent.

The guidance law commands, once per step, the impulse that puts a vehicle's
in-plane HCW velocity on a desired velocity: the target's velocity plus a
closing speed directed along the line of sight. The closing speed is the
smaller of a linear glideslope ``rho / slope_s + arrival_mps`` and the braking
curve ``sqrt(2 a_brake rho)`` that the vehicle's own per-step budget can shed,
where ``a_brake = brake_fraction * max_dv_mps / dt``. The resulting impulse is
scaled down to Euclidean norm ``max_dv_mps`` when it exceeds it.

The gain, ``slope_s`` and ``arrival_mps``, does not depend on ``max_dv_mps``:
the budget enters only through the braking curve and the final clip, both
physical limits of the vehicle. Two vehicles with different budgets therefore
command the same impulse wherever neither the braking curve nor the clip binds.

Defaults are ``slope_s = 300 s``, ``arrival_mps = 0.3 m/s`` and
``brake_fraction = 0.5``.

Reference: Hablani, Tapper, and Dana-Bashian, "Guidance and Relative Navigation
for Autonomous Rendezvous in a Circular Orbit," Journal of Guidance, Control,
and Dynamics 25(3), 2002, whose glideslope commands a range rate linear in
range so that the approach arrives at a chosen speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.policies.heuristic.views import IN_PLANE_RTN, mean_from_view
from orbitalgym.registry import PolicyKey, register


def _clip_to_cap(u: jax.Array, max_dv_mps: float) -> jax.Array:
    """Scale each row of ``u`` down to Euclidean norm ``max_dv_mps``, keeping its direction."""
    norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
    return u * jnp.minimum(1.0, max_dv_mps / jnp.maximum(norm, 1e-12))


def _glideslope_u(
    x: jax.Array,
    x_target: jax.Array,
    *,
    max_dv_mps: float,
    dt: float,
    slope_s: float,
    arrival_mps: float,
    brake_fraction: float,
) -> jax.Array:
    """The unclipped impulse that puts the in-plane velocity of ``x`` on the desired velocity."""
    rho_vec = x[..., :2] - x_target[..., :2]
    rho = jnp.linalg.norm(rho_vec, axis=-1, keepdims=True)
    rho_hat = rho_vec / jnp.maximum(rho, 1e-9)
    a_brake = brake_fraction * max_dv_mps / dt
    s_glide = rho / slope_s + arrival_mps
    s_brake = jnp.sqrt(2.0 * a_brake * rho)
    s = jnp.minimum(s_glide, s_brake)
    v_des = x_target[..., 2:] - s * rho_hat
    return v_des - x[..., 2:]


def glideslope_dv(
    x: jax.Array,
    x_target: jax.Array,
    *,
    max_dv_mps: float,
    dt: float,
    slope_s: float,
    arrival_mps: float,
    brake_fraction: float,
) -> jax.Array:
    """Glideslope impulse for in-plane states ``x``, ``x_target`` of shape ``(n, 4)``.

    Returns ``(n, 2)`` radial/along-track impulses of norm at most ``max_dv_mps``.
    """
    u = _glideslope_u(
        x,
        x_target,
        max_dv_mps=max_dv_mps,
        dt=dt,
        slope_s=slope_s,
        arrival_mps=arrival_mps,
        brake_fraction=brake_fraction,
    )
    return _clip_to_cap(u, max_dv_mps)


def _pad_and_wrap(dv_rt: jax.Array, command_cls: Any, n: int):
    """Zero-pad the in-plane impulse to the command's ``dv`` width and fill a zero command."""
    template = command_cls.zeros(n)
    dv_dim = template.dv.shape[-1]
    if dv_dim > 2:
        dv_rt = jnp.concatenate([dv_rt, jnp.zeros((n, dv_dim - 2), dv_rt.dtype)], axis=-1)
    return template.replace(dv=dv_rt.astype(template.dv.dtype))


@register(PolicyKey.GLIDESLOPE_TO_LADY)
@dataclass(frozen=True)
class GlideslopeToLady:
    """Glide onto the lady at the origin and push away from nearby opponents.

    The target state is the origin of the RTN frame, so the commanded velocity
    is the closing speed of the glideslope directed at the lady. Before the
    clip, a Gaussian repulsion ``avoidance_gain_mps * exp(-d^2 / (2 sigma^2))``
    per opponent, directed away from it, is added to the impulse.

    The agent view is either a belief with ``mean (N_obs, N_total, d)`` or a
    flat observation of the same numbers; observer ``i`` reads its own state
    from cell ``(i, i)`` and opponents from columns ``N_obs`` onward. The
    flat-observation path requires either a single full-state channel (e.g.
    FullObservation or ConicalObservation) or a
    :class:`~orbitalgym.observations.composite.CompositeObservation` whose
    channels are all full-state and the same size — in which case the last
    channel is used. Position-only channels must go through a belief.

    ``mean_motion`` is stored for interface symmetry with the other heuristic
    policies; the law itself does not use it.
    """

    mean_motion: float
    dt: float
    max_dv_mps: float
    slope_s: float
    arrival_mps: float
    brake_fraction: float
    avoidance_gain_mps: float
    avoidance_sigma_m: float
    n_vehicles: int
    n_opponents: int
    state_dim: int
    command_cls: Any

    @classmethod
    def build(
        cls,
        *,
        mean_motion: float,
        dt: float,
        n_vehicles: int,
        n_opponents: int,
        state_dim: int,
        command_cls: Any,
        max_dv_mps: float,
        slope_s: float = 300.0,
        arrival_mps: float = 0.3,
        brake_fraction: float = 0.5,
        avoidance_gain_mps: float = 0.0,
        avoidance_sigma_m: float = 50.0,
    ) -> GlideslopeToLady:
        return cls(
            mean_motion=float(mean_motion),
            dt=float(dt),
            max_dv_mps=float(max_dv_mps),
            slope_s=float(slope_s),
            arrival_mps=float(arrival_mps),
            brake_fraction=float(brake_fraction),
            avoidance_gain_mps=float(avoidance_gain_mps),
            avoidance_sigma_m=float(avoidance_sigma_m),
            n_vehicles=n_vehicles,
            n_opponents=n_opponents,
            state_dim=state_dim,
            command_cls=command_cls,
        )

    def _mean(self, agent_view: Any) -> jax.Array:
        return mean_from_view(agent_view, self.n_vehicles, self.n_opponents, self.state_dim)

    def __call__(self, policy_state: Any, agent_view: Any, key: jax.Array, t: jax.Array):
        del key, t
        mean = self._mean(agent_view)
        n = self.n_vehicles
        idx = jnp.arange(n)
        own = mean[idx, idx, :]
        own_rt = own[:, IN_PLANE_RTN] if self.state_dim == 6 else own
        opp_pos = mean[:, n:, :2]  # (n, n_opp, 2)

        u_glide = _glideslope_u(
            own_rt,
            jnp.zeros_like(own_rt),
            max_dv_mps=self.max_dv_mps,
            dt=self.dt,
            slope_s=self.slope_s,
            arrival_mps=self.arrival_mps,
            brake_fraction=self.brake_fraction,
        )
        diff = own_rt[:, None, :2] - opp_pos
        dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        direction = diff / (dist + 1e-9)
        magnitude = self.avoidance_gain_mps * jnp.exp(
            -(dist**2) / (2.0 * self.avoidance_sigma_m**2)
        )
        u_avoid = jnp.sum(magnitude * direction, axis=1)

        dv_rt = _clip_to_cap(u_glide + u_avoid, self.max_dv_mps)
        return _pad_and_wrap(dv_rt, self.command_cls, n), policy_state


@register(PolicyKey.GLIDESLOPE_INTERCEPT)
@dataclass(frozen=True)
class GlideslopeIntercept:
    """Glide onto the nearest opposing vehicle, matching its velocity as the range closes.

    Observer ``i`` picks the opposing vehicle nearest to it and takes that
    vehicle's in-plane state as the glideslope target, so the commanded
    velocity is the target's velocity plus the closing speed along the line of
    sight. The range and the relative speed therefore go to zero together.

    The agent view is read exactly as
    :class:`GlideslopeToLady` reads it.

    ``mean_motion`` is stored for interface symmetry with the other heuristic
    policies; the law itself does not use it.
    """

    mean_motion: float
    dt: float
    max_dv_mps: float
    slope_s: float
    arrival_mps: float
    brake_fraction: float
    n_vehicles: int
    n_opponents: int
    state_dim: int
    command_cls: Any

    @classmethod
    def build(
        cls,
        *,
        mean_motion: float,
        dt: float,
        n_vehicles: int,
        n_opponents: int,
        state_dim: int,
        command_cls: Any,
        max_dv_mps: float,
        slope_s: float = 300.0,
        arrival_mps: float = 0.3,
        brake_fraction: float = 0.5,
    ) -> GlideslopeIntercept:
        return cls(
            mean_motion=float(mean_motion),
            dt=float(dt),
            max_dv_mps=float(max_dv_mps),
            slope_s=float(slope_s),
            arrival_mps=float(arrival_mps),
            brake_fraction=float(brake_fraction),
            n_vehicles=n_vehicles,
            n_opponents=n_opponents,
            state_dim=state_dim,
            command_cls=command_cls,
        )

    def _mean(self, agent_view: Any) -> jax.Array:
        return mean_from_view(agent_view, self.n_vehicles, self.n_opponents, self.state_dim)

    def __call__(self, policy_state: Any, agent_view: Any, key: jax.Array, t: jax.Array):
        del key, t
        mean = self._mean(agent_view)
        n = self.n_vehicles
        idx = jnp.arange(n)
        own = mean[idx, idx, :]
        opp = mean[:, n:, :]  # (n, n_opp, d)
        if self.state_dim == 6:
            own_rt = own[:, IN_PLANE_RTN]
            opp_rt = opp[:, :, IN_PLANE_RTN]
        else:
            own_rt = own
            opp_rt = opp

        range_m = jnp.linalg.norm(own_rt[:, None, :2] - opp_rt[..., :2], axis=-1)  # (n, n_opp)
        nearest = jnp.argmin(range_m, axis=-1)  # (n,)
        target_rt = opp_rt[idx, nearest, :]  # (n, 4)

        dv_rt = glideslope_dv(
            own_rt,
            target_rt,
            max_dv_mps=self.max_dv_mps,
            dt=self.dt,
            slope_s=self.slope_s,
            arrival_mps=self.arrival_mps,
            brake_fraction=self.brake_fraction,
        )
        return _pad_and_wrap(dv_rt, self.command_cls, n), policy_state
