"""Finite-horizon HCW LQR toward the lady plus Gaussian repulsion from opponents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.registry import PolicyKey, register


def _hcw_rt_ab(mean_motion: float, dt: float) -> tuple[jax.Array, jax.Array]:
    """Discrete-time in-plane HCW (A, B) for state [r, t, r_dot, t_dot] and control [dv_r, dv_t]."""
    n = mean_motion
    s = jnp.sin(n * dt)
    c = jnp.cos(n * dt)
    phi = jnp.array(
        [
            [4 - 3 * c, 0.0, s / n, 2 * (1 - c) / n],
            [6 * (s - n * dt), 1.0, -2 * (1 - c) / n, (4 * s - 3 * n * dt) / n],
            [3 * n * s, 0.0, c, 2 * s],
            [-6 * n * (1 - c), 0.0, -2 * s, 4 * c - 3],
        ]
    )
    return phi, phi[:, 2:]


def _first_step_gain(mean_motion: float, dt: float, horizon: int, control_cost: float) -> jax.Array:
    """Gain of the first control in the unconstrained finite-horizon LQR to the origin."""
    a, b = _hcw_rt_ab(mean_motion, dt)
    powers = [jnp.eye(4)]
    for _ in range(horizon):
        powers.append(a @ powers[-1])
    m = jnp.concatenate([powers[horizon - 1 - k] @ b for k in range(horizon)], axis=1)
    c = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    h = m.T @ c.T @ c @ m + control_cost * jnp.eye(m.shape[1])
    k = jnp.linalg.solve(h, m.T @ c.T @ c @ powers[horizon])
    return k[:2, :]


IN_PLANE_RTN = jnp.array([0, 1, 3, 4])


@register(PolicyKey.LQR_AVOID)
@dataclass(frozen=True)
class LQRGoToLadyWithAvoidance:
    """Drive to the lady with an in-plane LQR and push away from nearby opponents.

    The control is ``clip(u_lqr + u_avoid, ±max_dv_mps)`` where ``u_avoid`` sums
    ``avoidance_gain * exp(-d^2 / (2 sigma^2))`` per opponent, directed away from
    it. The agent view is either a belief with ``mean (N_obs, N_total, d)`` or a
    flat observation of the same numbers; observer ``i`` reads its own state
    from cell ``(i, i)`` and opponents from columns ``N_obs`` onward. The
    flat-observation path requires either a single full-state channel (e.g.
    FullObservation or ConicalObservation) or a
    :class:`~orbitalgym.observations.composite.CompositeObservation` whose
    channels are all full-state and the same size — in which case the last
    channel is used (e.g. teammate-ephemeris + conical, where the conical
    channel carries the truth-anchored per-pair state this policy needs).
    Position-only channels must go through a belief.
    """

    gain: jax.Array
    max_dv_mps: float
    avoidance_gain: float
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
        horizon: int = 8,
        control_cost: float = 1e-3,
        max_dv_mps: float = 0.5,
        avoidance_gain: float = 1.0,
        avoidance_sigma_m: float = 300.0,
    ) -> LQRGoToLadyWithAvoidance:
        return cls(
            gain=_first_step_gain(mean_motion, dt, horizon, control_cost),
            max_dv_mps=float(max_dv_mps),
            avoidance_gain=float(avoidance_gain),
            avoidance_sigma_m=float(avoidance_sigma_m),
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
                # Multiple equal-sized full-state channels concatenated by
                # CompositeObservation: use the last one.
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
        own_rt = own[:, IN_PLANE_RTN] if self.state_dim == 6 else own
        opp_pos = mean[:, n:, :2]  # (n, n_opp, 2)

        u_lqr = -own_rt @ self.gain.T  # (n, 2)
        diff = own_rt[:, None, :2] - opp_pos
        dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        direction = diff / (dist + 1e-9)
        magnitude = self.avoidance_gain * jnp.exp(-(dist**2) / (2.0 * self.avoidance_sigma_m**2))
        u_avoid = jnp.sum(magnitude * direction, axis=1)

        dv_rt = jnp.clip(u_lqr + u_avoid, -self.max_dv_mps, self.max_dv_mps)
        template = self.command_cls.zeros(n)
        dv_dim = template.dv.shape[-1]
        if dv_dim > 2:
            dv_rt = jnp.concatenate([dv_rt, jnp.zeros((n, dv_dim - 2), dv_rt.dtype)], axis=-1)
        return template.replace(dv=dv_rt.astype(template.dv.dtype)), policy_state
