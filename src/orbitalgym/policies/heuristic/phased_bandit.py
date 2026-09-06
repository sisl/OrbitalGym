"""Scheduled bandit: coast, transfer onto a standoff ring, hold there, then commit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from orbitalgym.policies.heuristic.glideslope import GlideslopeToLady, glideslope_dv
from orbitalgym.policies.heuristic.views import IN_PLANE_RTN, mean_from_view
from orbitalgym.registry import PolicyKey, register

COAST = 0
TRANSFER = 1
HOLD = 2
COMMIT = 3


@flax.struct.dataclass
class PhasedBanditState:
    """Per-vehicle phase index and time spent in the current phase.

    ``phase`` holds one of ``COAST``, ``TRANSFER``, ``HOLD``, ``COMMIT`` per
    vehicle; each vehicle advances on its own, so a fleet whose members settle
    onto the ring at different times still spends the full hold on it.
    ``t_in_phase`` counts seconds since the vehicle entered its current phase
    and resets on every transition.
    """

    phase: jax.Array  # (n_vehicles,) int32
    t_in_phase: jax.Array  # (n_vehicles,) float32


def _ring_target(own_rt: jax.Array, standoff_m: float, mean_motion: float) -> jax.Array:
    """The state on the natural 2:1 ellipse of radial amplitude ``standoff_m``.

    The ellipse family is the one
    :class:`~orbitalgym.sampling.side.RelativeEllipse` samples: radial
    amplitude ``A`` with along-track amplitude ``2A``, centred on the lady and
    free of secular drift, traced as ``r = -A cos(theta)``,
    ``t = 2A sin(theta)`` at ``theta`` advancing at the mean motion. The target
    sits at the vehicle's own current phase angle, read off its position as
    ``cos(theta) = -r / rho`` and ``sin(theta) = (t / 2) / rho`` with
    ``rho = sqrt(r^2 + (t/2)^2)``, so the transfer is a change of amplitude
    rather than of phase. The target velocity is the ellipse's own velocity at
    that phase, so a vehicle that reaches the target coasts around the ring.

    A vehicle exactly on the lady has no defined phase; the radial direction is
    used there so the target is still a point on the ring.
    """
    radial = own_rt[:, 0]
    along = own_rt[:, 1]
    half_along = 0.5 * along
    rho = jnp.sqrt(radial**2 + half_along**2)
    degenerate = rho < 1e-6
    u_r = jnp.where(degenerate, 1.0, radial / jnp.maximum(rho, 1e-12))
    u_t = jnp.where(degenerate, 0.0, half_along / jnp.maximum(rho, 1e-12))
    amp = standoff_m
    return jnp.stack(
        [
            amp * u_r,
            2.0 * amp * u_t,
            mean_motion * amp * u_t,
            -2.0 * mean_motion * amp * u_r,
        ],
        axis=-1,
    )


@register(PolicyKey.PHASED_BANDIT)
@dataclass(frozen=True)
class PhasedBandit:
    """Run a bandit through a coast, a transfer to a standoff ring, a hold, and a commit.

    The schedule makes timing and information matter: the bandit is quiet for
    ``t_coast_s``, spends fuel once to move onto a ring of radial amplitude
    ``standoff_m``, waits there for ``t_hold_s``, and only then closes on the
    lady.

    1. ``coast`` — zero delta-v for ``t_coast_s`` seconds.
    2. ``transfer`` — glide onto the natural 2:1 ellipse of radial amplitude
       ``standoff_m`` at the vehicle's current phase angle (see
       :func:`_ring_target`), using
       :func:`~orbitalgym.policies.heuristic.glideslope.glideslope_dv` with the
       commit policy's ``slope_s`` and ``brake_fraction`` and an arrival speed
       of zero, so the vehicle arrives on the ring state rather than through
       it. Ends when the state is within ``settle_tol_m`` in position and
       ``settle_tol_mps`` in velocity of that target, or after
       ``t_transfer_max_s`` seconds.
    3. ``hold`` — zero delta-v for ``t_hold_s`` seconds. The ring is a bounded
       relative orbit with no secular drift, so holding costs no fuel; a hold
       at a fixed RTN point would have to be paid for every step.
    4. ``commit`` — the
       :class:`~orbitalgym.policies.heuristic.glideslope.GlideslopeToLady`
       command, with avoidance as configured, until the episode ends.

    Phases are per-vehicle and live in ``policy_state``; transitions are
    ``jnp.where`` selects on traced values, so the policy runs under ``jit``
    and ``vmap``. Phase time is counted in ``dt`` increments from the first
    call rather than read off the absolute ``t``, so the schedule is measured
    from the start of the episode.

    ``settle_tol_mps`` decides how well the hold holds. A residual velocity
    error ``dv`` violates the bounded-orbit condition and leaves an along-track
    drift of about ``2 pi dv / (1.5 n)`` metres per orbit, so how tightly the
    transfer ends sets how far the vehicle walks off the ring during the very
    phase the schedule exists to create. Measured on a 2 km standoff ring
    approached from 3 km at a 0.4545 m/s per-step cap, over six start phases,
    ``settle_tol_mps=0.2`` settles in 510 to 630 s and holds the amplitude
    within 47 to 75 percent of the standoff over an orbit. The default 0.02
    settles in 750 to 840 s and holds within 3.4 to 7.7 percent, so the hold
    stays near the ring rather than depending on where in the ellipse the
    looser threshold happens to stop the transfer.

    The transfer carries no avoidance term by design. Pushing away from a guard
    while gliding onto the ring would fight the ring target and leave the
    vehicle short of it when the phase ends. Avoidance acts in the commit
    phase, through the wrapped policy's ``avoidance_gain_mps`` and
    ``avoidance_sigma_m``.

    Every phase's command is capped at ``max_dv_mps`` by the norm clip the
    glideslope law applies. The agent view is a belief with
    ``mean (N_obs, N_total, d)`` or a flat observation of the same numbers, as
    :func:`~orbitalgym.policies.heuristic.views.mean_from_view` describes.
    """

    commit: GlideslopeToLady
    mean_motion: float
    dt: float
    t_coast_s: float
    standoff_m: float
    t_hold_s: float
    settle_tol_m: float
    settle_tol_mps: float
    t_transfer_max_s: float

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
        t_coast_s: float,
        standoff_m: float,
        t_hold_s: float,
        settle_tol_m: float = 20.0,
        settle_tol_mps: float = 0.02,
        t_transfer_max_s: float = 3000.0,
        slope_s: float = 100.0,
        arrival_mps: float = 0.3,
        brake_fraction: float = 0.5,
        avoidance_gain_mps: float = 0.0,
        avoidance_sigma_m: float = 50.0,
    ) -> PhasedBandit:
        return cls(
            commit=GlideslopeToLady.build(
                mean_motion=mean_motion,
                dt=dt,
                n_vehicles=n_vehicles,
                n_opponents=n_opponents,
                state_dim=state_dim,
                command_cls=command_cls,
                max_dv_mps=max_dv_mps,
                slope_s=slope_s,
                arrival_mps=arrival_mps,
                brake_fraction=brake_fraction,
                avoidance_gain_mps=avoidance_gain_mps,
                avoidance_sigma_m=avoidance_sigma_m,
            ),
            mean_motion=float(mean_motion),
            dt=float(dt),
            t_coast_s=float(t_coast_s),
            standoff_m=float(standoff_m),
            t_hold_s=float(t_hold_s),
            settle_tol_m=float(settle_tol_m),
            settle_tol_mps=float(settle_tol_mps),
            t_transfer_max_s=float(t_transfer_max_s),
        )

    def init_state(self) -> PhasedBanditState:
        """Every vehicle starts coasting with its phase clock at zero."""
        n = self.commit.n_vehicles
        return PhasedBanditState(
            phase=jnp.zeros((n,), dtype=jnp.int32),
            t_in_phase=jnp.zeros((n,), dtype=jnp.float32),
        )

    def __call__(
        self,
        policy_state: PhasedBanditState,
        agent_view: Any,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, PhasedBanditState]:
        inner = self.commit
        mean = mean_from_view(agent_view, inner.n_vehicles, inner.n_opponents, inner.state_dim)
        n = inner.n_vehicles
        idx = jnp.arange(n)
        own = mean[idx, idx, :]
        own_rt = own[:, IN_PLANE_RTN] if inner.state_dim == 6 else own

        target = _ring_target(own_rt, self.standoff_m, self.mean_motion)
        error = own_rt - target
        dv_transfer = glideslope_dv(
            own_rt,
            target,
            max_dv_mps=inner.max_dv_mps,
            dt=self.dt,
            slope_s=inner.slope_s,
            arrival_mps=0.0,
            brake_fraction=inner.brake_fraction,
        )

        commit_command, _ = inner(None, agent_view, key, t)
        dv_commit = commit_command.dv
        dv_dim = dv_commit.shape[-1]
        if dv_dim > 2:
            dv_transfer = jnp.concatenate(
                [dv_transfer, jnp.zeros((n, dv_dim - 2), dv_transfer.dtype)], axis=-1
            )

        phase = policy_state.phase
        in_transfer = (phase == TRANSFER)[:, None]
        in_commit = (phase == COMMIT)[:, None]
        dv = jnp.where(in_transfer, dv_transfer, 0.0) + jnp.where(in_commit, dv_commit, 0.0)

        t_next = policy_state.t_in_phase + self.dt
        settled = jnp.logical_and(
            jnp.linalg.norm(error[:, :2], axis=-1) < self.settle_tol_m,
            jnp.linalg.norm(error[:, 2:], axis=-1) < self.settle_tol_mps,
        )
        advance = jnp.where(
            phase == COAST,
            t_next >= self.t_coast_s,
            jnp.where(
                phase == TRANSFER,
                jnp.logical_or(settled, t_next >= self.t_transfer_max_s),
                jnp.where(phase == HOLD, t_next >= self.t_hold_s, False),
            ),
        )
        new_state = PhasedBanditState(
            phase=phase + advance.astype(phase.dtype),
            t_in_phase=jnp.where(advance, 0.0, t_next).astype(policy_state.t_in_phase.dtype),
        )
        return commit_command.replace(dv=dv.astype(dv_commit.dtype)), new_state
