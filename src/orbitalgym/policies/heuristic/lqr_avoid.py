"""Infinite-horizon HCW LQR toward the lady plus Gaussian repulsion from opponents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are

from orbitalgym.registry import PolicyKey, register


def _hcw_rt_ab(mean_motion: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time in-plane HCW (A, B) for state [r, t, r_dot, t_dot] and control [dv_r, dv_t]."""
    n = float(mean_motion)
    s = np.sin(n * dt)
    c = np.cos(n * dt)
    phi = np.array(
        [
            [4 - 3 * c, 0.0, s / n, 2 * (1 - c) / n],
            [6 * (s - n * dt), 1.0, -2 * (1 - c) / n, (4 * s - 3 * n * dt) / n],
            [3 * n * s, 0.0, c, 2 * s],
            [-6 * n * (1 - c), 0.0, -2 * s, 4 * c - 3],
        ],
        dtype=np.float64,
    )
    return phi, phi[:, 2:]


def _steady_state_gain(
    mean_motion: float, dt: float, r_scale_m: float, v_scale_mps: float, dv_scale_mps: float
) -> jax.Array:
    """Infinite-horizon LQR gain from the float64 discrete algebraic Riccati equation."""
    a, b = _hcw_rt_ab(mean_motion, dt)
    q = np.diag(
        np.array(
            [1.0 / r_scale_m**2, 1.0 / r_scale_m**2, 1.0 / v_scale_mps**2, 1.0 / v_scale_mps**2],
            dtype=np.float64,
        )
    )
    r = np.eye(2, dtype=np.float64) / dv_scale_mps**2
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return jnp.asarray(gain, dtype=jnp.float32)


IN_PLANE_RTN = jnp.array([0, 1, 3, 4])


def mean_from_view(agent_view: Any, n_vehicles: int, n_opponents: int, state_dim: int) -> jax.Array:
    """The ``(N_obs, N_total, d)`` state block of a belief view or a flat observation.

    A belief exposes it as ``.mean``. A flat observation carries the same
    numbers in row-major order; when it holds several equal-sized full-state
    channels (as :class:`~orbitalgym.observations.composite.CompositeObservation`
    produces) the last channel is used. Any other size is rejected, because a
    position-only channel cannot supply the velocities these policies regulate.
    """
    if not isinstance(agent_view, jax.Array):
        return agent_view.mean
    n_total = n_vehicles + n_opponents
    expected_size = n_vehicles * n_total * state_dim
    size = agent_view.size
    if size == expected_size:
        flat = agent_view
    elif size % expected_size == 0:
        flat = agent_view.reshape((-1, expected_size))[-1]
    else:
        raise ValueError(
            f"Flat observation size {size} is not a multiple of expected "
            f"{expected_size} ({n_vehicles} vehicles × {n_total} entities × "
            f"{state_dim} dims). Flat-observation path requires one or more "
            f"equal-sized full-state channels; position-only channels must use a "
            f"belief view."
        )
    return flat.reshape((n_vehicles, n_total, state_dim))


@register(PolicyKey.LQR_AVOID)
@dataclass(frozen=True)
class LQRGoToLadyWithAvoidance:
    """Drive to the lady with an in-plane LQR and push away from nearby opponents.

    ``u_lqr = -K x`` uses the infinite-horizon gain ``K`` of the in-plane HCW
    model, obtained at build time from the discrete algebraic Riccati equation
    solved in float64 and stored as float32. ``Q`` penalises position by
    ``1 / r_scale_m^2`` and velocity by ``1 / v_scale_mps^2``; ``R`` penalises
    control by ``1 / dv_scale_mps^2``.

    ``u_avoid`` sums ``avoidance_gain * exp(-d^2 / (2 sigma^2))`` per opponent,
    directed away from it. The combined command ``u_lqr + u_avoid`` is scaled
    down to Euclidean norm ``max_dv_mps`` when it exceeds it, preserving its
    direction, so the command never exceeds the per-step budget the environment
    enforces. The agent view is either a belief with ``mean (N_obs, N_total, d)`` or a
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
        max_dv_mps: float = 0.5,
        r_scale_m: float = 100.0,
        v_scale_mps: float = 1.0,
        dv_scale_mps: float | None = None,
        avoidance_gain: float = 1.0,
        avoidance_sigma_m: float = 300.0,
    ) -> LQRGoToLadyWithAvoidance:
        dv_scale = float(max_dv_mps) if dv_scale_mps is None else float(dv_scale_mps)
        return cls(
            gain=_steady_state_gain(
                mean_motion, dt, float(r_scale_m), float(v_scale_mps), dv_scale
            ),
            max_dv_mps=float(max_dv_mps),
            avoidance_gain=float(avoidance_gain),
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

        u_lqr = -own_rt @ self.gain.T  # (n, 2)
        diff = own_rt[:, None, :2] - opp_pos
        dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        direction = diff / (dist + 1e-9)
        magnitude = self.avoidance_gain * jnp.exp(-(dist**2) / (2.0 * self.avoidance_sigma_m**2))
        u_avoid = jnp.sum(magnitude * direction, axis=1)

        u = u_lqr + u_avoid
        u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
        dv_rt = u * jnp.minimum(1.0, self.max_dv_mps / jnp.maximum(u_norm, 1e-12))
        template = self.command_cls.zeros(n)
        dv_dim = template.dv.shape[-1]
        if dv_dim > 2:
            dv_rt = jnp.concatenate([dv_rt, jnp.zeros((n, dv_dim - 2), dv_rt.dtype)], axis=-1)
        return template.replace(dv=dv_rt.astype(template.dv.dtype)), policy_state
