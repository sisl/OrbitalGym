"""Infinite-horizon HCW LQR driving the relative state to the nearest opponent to zero."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.policies.heuristic.lqr_avoid import IN_PLANE_RTN, _steady_state_gain
from orbitalgym.registry import PolicyKey, register


@register(PolicyKey.LQR_INTERCEPT)
@dataclass(frozen=True)
class LQRIntercept:
    """Close on the nearest opposing vehicle with an in-plane LQR on the relative state.

    Observer ``i`` forms ``x = own - target``, where ``target`` is the mean
    position and velocity of the opposing vehicle nearest to it, and commands
    ``u = -K x``. The HCW equations are linear, so the difference of two HCW
    states obeys the same in-plane HCW dynamics with the difference of the two
    controls as input; the target's own thrust enters only as a disturbance.
    The gain ``K`` computed for the absolute state therefore also stabilises
    the relative state, and this policy reuses the infinite-horizon gain of
    :class:`~orbitalgym.policies.heuristic.lqr_avoid.LQRGoToLadyWithAvoidance`,
    obtained at build time from the discrete algebraic Riccati equation solved
    in float64 and stored as float32. ``Q`` penalises relative position by
    ``1 / r_scale_m^2`` and relative velocity by ``1 / v_scale_mps^2``; ``R``
    penalises control by ``1 / dv_scale_mps^2``.

    The command is scaled down to Euclidean norm ``max_dv_mps`` when it exceeds
    it, preserving its direction, so it never exceeds the per-step budget the
    environment enforces. The agent view is either a belief with
    ``mean (N_obs, N_total, d)`` or a flat observation of the same numbers;
    observer ``i`` reads its own state from cell ``(i, i)`` and opponents from
    columns ``N_obs`` onward, exactly as ``LQRGoToLadyWithAvoidance`` does. The
    flat-observation path requires either a single full-state channel (e.g.
    FullObservation or ConicalObservation) or a
    :class:`~orbitalgym.observations.composite.CompositeObservation` whose
    channels are all full-state and the same size — in which case the last
    channel is used. Position-only channels must go through a belief.
    """

    gain: jax.Array
    max_dv_mps: float
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
        r_scale_m: float = 100.0,
        v_scale_mps: float = 1.0,
        dv_scale_mps: float | None = None,
    ) -> LQRIntercept:
        dv_scale = float(max_dv_mps) if dv_scale_mps is None else float(dv_scale_mps)
        return cls(
            gain=_steady_state_gain(
                mean_motion, dt, float(r_scale_m), float(v_scale_mps), dv_scale
            ),
            max_dv_mps=float(max_dv_mps),
            n_vehicles=n_vehicles,
            n_opponents=n_opponents,
            state_dim=state_dim,
            command_cls=command_cls,
        )

    def _mean(self, agent_view: Any) -> jax.Array:
        if isinstance(agent_view, jax.Array):
            n_total = self.n_vehicles + self.n_opponents
            expected_size = self.n_vehicles * n_total * self.state_dim
            size = agent_view.size
            if size == expected_size:
                flat = agent_view
            elif size % expected_size == 0:
                flat = agent_view.reshape((-1, expected_size))[-1]
            else:
                raise ValueError(
                    f"Flat observation size {size} is not a multiple of expected "
                    f"{expected_size} ({self.n_vehicles} vehicles × {n_total} entities × "
                    f"{self.state_dim} dims). Flat-observation path requires one or more "
                    f"equal-sized full-state channels; position-only channels must use a "
                    f"belief view."
                )
            return flat.reshape((self.n_vehicles, n_total, self.state_dim))
        return agent_view.mean

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

        u = -(own_rt - target_rt) @ self.gain.T
        u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
        dv_rt = u * jnp.minimum(1.0, self.max_dv_mps / jnp.maximum(u_norm, 1e-12))
        template = self.command_cls.zeros(n)
        dv_dim = template.dv.shape[-1]
        if dv_dim > 2:
            dv_rt = jnp.concatenate([dv_rt, jnp.zeros((n, dv_dim - 2), dv_rt.dtype)], axis=-1)
        return template.replace(dv=dv_rt.astype(template.dv.dtype)), policy_state
