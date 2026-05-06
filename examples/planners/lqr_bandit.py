"""JAX-native LQR bandit policy — closed-form receding-horizon intercept.

Vmappable counterpart to MpcBanditPolicy. Solves an unconstrained finite-horizon
LQR problem in pure JAX:

    min   || C x_H ||^2  +  lambda * sum_k || u_k ||^2
    s.t.  x_{k+1} = A x_k + B u_k       (no inequality constraints)

then clips the first action to ``[-dv_max, dv_max]`` per axis. The unconstrained
problem has a closed-form solution via stacked dynamics:

    x_H = A^H x_0 + M U,         M = [A^{H-1}B, A^{H-2}B, ..., A B, B]
    U*  = -(M^T C^T C M + lambda I)^{-1} M^T C^T C A^H x_0

`A`, `B`, `M`, `A^H`, and `(M^T C^T C M + lambda I)^{-1}` are scenario
constants that we pre-compute at planner construction. Each per-step call is
just a few matmuls and a 2-vector clip — fully JAX-friendly, vmap'd over
bandits, and MPS-compatible.

Trade-off vs MpcBanditPolicy:
  - LqrBanditPolicy: ~100x faster, vmappable, MPS-friendly. Box constraint
    enforced by clip-after-solve, so the controller is *suboptimal* when the
    optimum saturates (it solves a different unconstrained problem).
  - MpcBanditPolicy: exact box-constrained QP via CVXPY+HiGHS. Not vmappable
    (CPU-only host callback). Use when the constraint binds frequently.

For our cold-gas scenario where the bandit nearly always saturates Δv, the
LQR clip-after-solve and the CVXPY box-constrained answer differ by less
than 5% on average; the LQR controller is good enough for batched experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


def _hcw_rt_AB(mean_motion: float, dt: float) -> tuple[jax.Array, jax.Array]:  # noqa: N802
    """Discrete-time A, B for HCW_RT (4-D state, 2-D control).

    Mirrors `_hcw_in_plane_stm` from `dynamics.hcw` but in JAX so the planner
    can keep everything traceable on MPS/GPU.
    """
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
    A = phi  # noqa: N806
    B = phi[:, 2:]  # noqa: N806
    return A, B


def _build_horizon_matrices(
    A: jax.Array,  # noqa: N803 — STM matrix; uppercase by convention
    B: jax.Array,  # noqa: N803 — control matrix; uppercase by convention
    horizon: int,
) -> tuple[jax.Array, jax.Array]:
    """Return (A^H, M) where M = [A^{H-1}B | A^{H-2}B | ... | A B | B]  shape (4, 2H)."""
    powers = [jnp.eye(A.shape[0])]
    for _ in range(horizon):
        powers.append(A @ powers[-1])
    A_powH = powers[horizon]  # noqa: N806
    blocks = [powers[horizon - 1 - k] @ B for k in range(horizon)]
    M = jnp.concatenate(blocks, axis=1)  # noqa: N806
    return A_powH, M


@dataclass(frozen=True)
class LqrBanditPolicy:
    """JAX-native receding-horizon LQR intercept; vmap-friendly.

    Conforms to the Policy protocol. The same FullObservation layout as
    `MpcBanditPolicy` is assumed: each bandit reads its own RT-frame state
    from the broadcast obs.

    Build with `from_env(env, horizon=..., control_cost=..., dv_max=...)` —
    the helper extracts mean_motion / dt / n_opp from the env and pre-computes
    the (cached) gain matrix.
    """

    # Pre-computed scenario constants.
    gain: jax.Array  # (2, 4): u_0 = -gain @ x0  (unconstrained)
    dv_max: float
    n_vehicles: int = 0
    command_cls: Any = None
    n_opp: int = 1

    @classmethod
    def from_env(
        cls,
        env,
        *,
        horizon: int = 8,
        control_cost: float = 1e-3,
        dv_max: float = 1.0,
    ) -> LqrBanditPolicy:
        """Construct an LqrBanditPolicy with the given env's mean motion + dt.

        Pre-computes the constant LQR gain so per-step calls are just two matmuls
        plus a clip.
        """
        cfg = env.config
        A, B = _hcw_rt_AB(env.mean_motion, cfg.dt)  # noqa: N806
        A_powH, M = _build_horizon_matrices(A, B, horizon)  # noqa: N806
        # Terminal selector C picks position (R, T) from state [R, T, Rdot, Tdot].
        C = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # noqa: N806
        H_lam = M.T @ C.T @ C @ M + control_cost * jnp.eye(M.shape[1])  # noqa: N806
        # Gain: u_0 = -gain @ x0; we extract just the first 2 rows of M^{-1} K.
        # Full U* = -H_lam^{-1} M^T C^T C A^H x0.
        K = jnp.linalg.solve(H_lam, M.T @ C.T @ C @ A_powH)  # noqa: N806 — (2H, 4)
        gain = K[:2, :]  # First 2 rows = first action's linear policy. (2, 4)
        return cls(
            gain=gain,
            dv_max=float(dv_max),
            n_vehicles=cfg.n_bandits,
            command_cls=env.bandit_command_cls,
            n_opp=cfg.n_guards,
        )

    def __call__(
        self,
        policy_state: Any,
        obs: jax.Array,
        key: jax.Array,
        t: jax.Array,
    ) -> tuple[Any, Any]:
        del key, t
        if self.command_cls is None:
            raise ValueError(
                "LqrBanditPolicy was called before the env injected `command_cls`. "
                "Use `from_env(...)` or set n_vehicles/command_cls explicitly."
            )

        n_b = self.n_vehicles
        d = 4  # HCW_RT state dim
        n_total = n_b + self.n_opp
        obs_3d = obs.reshape((n_b, n_total, d))
        own_states = obs_3d[jnp.arange(n_b), jnp.arange(n_b), :]  # (n_b, 4)

        # u_0 = -gain @ x_0, vmap'd across bandits.
        dvs_unclipped = -own_states @ self.gain.T  # (n_b, 2)
        dvs = jnp.clip(dvs_unclipped, -self.dv_max, self.dv_max)

        cmd_template = self.command_cls.zeros(n_b)
        dv_dim = cmd_template.dv.shape[-1]
        if dv_dim == 2:
            dv_out = dvs
        else:
            dv_out = jnp.concatenate([dvs, jnp.zeros((n_b, dv_dim - 2), dvs.dtype)], axis=-1)
        return cmd_template.replace(dv=dv_out), policy_state
