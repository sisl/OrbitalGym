"""JAX-native glideslope bandit policy — saturation-aware approach to the lady.

Wraps :func:`orbitalgym.policies.heuristic.glideslope.glideslope_dv` in the
flat-observation contract the ring-intercept example uses. Each bandit reads
its own in-plane state from the broadcast observation and commands the impulse
that puts its velocity on the glideslope toward the RTN origin:

    s     = min(rho / slope_s + arrival_mps, sqrt(2 a_brake rho))
    dv    = clip_norm(-s * rho_hat - v, dv_max)

with ``a_brake = brake_fraction * dv_max / dt``. The gain, ``slope_s`` and
``arrival_mps``, is independent of ``dv_max``: the budget enters only through
the braking curve and the norm clip, both physical limits of the vehicle. The
per-step call is a handful of elementwise operations, so the policy is
traceable, vmappable, and accelerator-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.policies.heuristic.glideslope import glideslope_dv


@dataclass(frozen=True)
class GlideslopeBanditPolicy:
    """Glideslope approach to the lady; vmap-friendly.

    Conforms to the Policy protocol. The same FullObservation layout as
    `MPCBanditPolicy` is assumed: each bandit reads its own RT-frame state
    from the broadcast obs.

    Build with `from_env(env, slope_s=..., arrival_mps=..., dv_max=...)` — the
    helper extracts dt and the vehicle counts from the env.
    """

    dt: float
    slope_s: float
    arrival_mps: float
    brake_fraction: float
    dv_max: float
    n_vehicles: int = 0
    command_cls: Any = None
    n_opp: int = 1

    @classmethod
    def from_env(
        cls,
        env,
        *,
        slope_s: float = 300.0,
        arrival_mps: float = 0.3,
        brake_fraction: float = 0.5,
        dv_max: float = 1.0,
    ) -> GlideslopeBanditPolicy:
        """Construct a GlideslopeBanditPolicy with the given env's dt and vehicle counts."""
        cfg = env.config
        return cls(
            dt=float(cfg.dt),
            slope_s=float(slope_s),
            arrival_mps=float(arrival_mps),
            brake_fraction=float(brake_fraction),
            dv_max=float(dv_max),
            n_vehicles=cfg.n_bandits,
            command_cls=env.bandit_command_cls,
            n_opp=cfg.n_guards,
        )

    def __call__(
        self,
        policy_state: Any,
        agent_view: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        del key, t
        if self.command_cls is None:
            raise ValueError(
                "GlideslopeBanditPolicy was called before the env injected `command_cls`. "
                "Use `from_env(...)` or set n_vehicles/command_cls explicitly."
            )

        n_b = self.n_vehicles
        d = 4  # HCW_RT state dim
        n_total = n_b + self.n_opp
        obs_3d = agent_view.reshape((n_b, n_total, d))
        own_states = obs_3d[jnp.arange(n_b), jnp.arange(n_b), :]  # (n_b, 4)

        dvs = glideslope_dv(
            own_states,
            jnp.zeros_like(own_states),
            max_dv_mps=self.dv_max,
            dt=self.dt,
            slope_s=self.slope_s,
            arrival_mps=self.arrival_mps,
            brake_fraction=self.brake_fraction,
        )

        cmd_template = self.command_cls.zeros(n_b)
        dv_dim = cmd_template.dv.shape[-1]
        if dv_dim == 2:
            dv_out = dvs
        else:
            dv_out = jnp.concatenate([dvs, jnp.zeros((n_b, dv_dim - 2), dvs.dtype)], axis=-1)
        return cmd_template.replace(dv=dv_out), policy_state
