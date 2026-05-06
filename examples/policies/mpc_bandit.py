"""MPC bandit policy — receding-horizon QP intercept of the lady.

Each bandit independently solves a finite-horizon convex QP at every step:

    min   || x_H[:2] ||^2  +  lambda * sum_k || u_k ||^2
    s.t.  x_{k+1} = A x_k + B u_k,        k = 0, ..., H-1
          | u_k[j] | <= dv_max,            j = 0, 1

with linear HCW_RT dynamics (4-D state [r, t, rdot, tdot], 2-D control [dv_r,
dv_t]). The first control u_0 is applied; the rest is discarded (receding
horizon). Solved with CVXPY + HiGHS via `jax.pure_callback`, so it slots into
the package's Policy protocol but is *not* vmappable.

The cold-gas Moog thruster (3.6 N) sets dv_max via
    dv_max = max_thrust_n * dt / wet_mass_kg

We use the *initial* wet mass (params.dry_mass_kg + propellant tank size) for
the bound — under-conservative as propellant burns off, but the residual
propellant constraint is enforced separately by the impulsive maneuver
component (it clamps mass to >= 0 via the rocket equation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# CVXPY + HiGHS are runtime requirements for this planner only.
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np


def _build_hcw_rt_matrices(mean_motion: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the discrete-time A, B for HCW_RT (in-plane HCW) over one step.

    Mirrors `_hcw_in_plane_stm` in `orbital_game.dynamics.hcw` but in NumPy so
    CVXPY can use it as constant problem data. State: [r, t, rdot, tdot].
    Control u = [dv_r, dv_t] is applied as an impulse at the start of the step.

    The package's step is x_{k+1} = (x_k + [0, 0, dv]) @ Phi.T (row form), which
    is the same as x_{k+1} = Phi @ x_k + Phi[:, 2:] @ dv (column form). So:
        A = Phi
        B = Phi[:, 2:]
    """
    n = mean_motion
    s = np.sin(n * dt)
    c = np.cos(n * dt)
    phi = np.array(
        [
            [4 - 3 * c, 0.0, s / n, 2 * (1 - c) / n],
            [6 * (s - n * dt), 1.0, -2 * (1 - c) / n, (4 * s - 3 * n * dt) / n],
            [3 * n * s, 0.0, c, 2 * s],
            [-6 * n * (1 - c), 0.0, -2 * s, 4 * c - 3],
        ]
    )
    A = phi  # noqa: N806 — STM matrix; uppercase by convention
    B = phi[:, 2:]  # noqa: N806 — control matrix; uppercase by convention
    return A, B


def _solve_one_bandit(
    x0: np.ndarray,
    A: np.ndarray,  # noqa: N803 — STM matrix; uppercase by convention
    B: np.ndarray,  # noqa: N803 — control matrix; uppercase by convention
    horizon: int,
    dv_max: float,
    control_cost: float,
) -> np.ndarray:
    """Solve the per-bandit QP. Returns the first control u_0 of shape (2,).

    On solver failure (rare for well-posed inputs), returns zeros — the bandit
    coasts that step.
    """
    U = cp.Variable((horizon, 2))  # noqa: N806 — decision-variable matrix
    x = x0.astype(np.float64)
    cost = 0.0
    constraints = []
    for k in range(horizon):
        x = A @ x + B @ U[k]
        constraints.append(cp.abs(U[k]) <= dv_max)
        cost = cost + control_cost * cp.sum_squares(U[k])
    # Terminal position-only cost: || x_H[:2] ||^2.
    cost = cost + cp.sum_squares(x[:2])
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.HIGHS)
    except cp.error.SolverError:
        return np.zeros(2, dtype=np.float64)
    if U.value is None:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(U.value[0], dtype=np.float64)


def _solve_batch(
    own_states: np.ndarray,
    A: np.ndarray,  # noqa: N803 — STM matrix; uppercase by convention
    B: np.ndarray,  # noqa: N803 — control matrix; uppercase by convention
    horizon: int,
    dv_max: float,
    control_cost: float,
) -> np.ndarray:
    """Solve a batch of independent per-bandit QPs sequentially."""
    n_b = own_states.shape[0]
    out = np.zeros((n_b, 2), dtype=np.float64)
    for i in range(n_b):
        out[i] = _solve_one_bandit(own_states[i], A, B, horizon, dv_max, control_cost)
    return out


@dataclass(frozen=True)
class MPCBanditPolicy:
    """Receding-horizon QP intercept policy, one QP per bandit per step.

    Conforms to the Policy protocol (callable as
    `(policy_state, obs, key, t) -> (command, next_policy_state)`). The env
    populates `n_vehicles` and `command_cls` at construction. The `obs` is a
    flat FullObservation: per-bandit views are broadcast `(n_b, n_total, 4)`,
    flattened to a single 1-D array. We reshape and extract each bandit's own
    state (the i-th row of own-block in observer i's view).

    The CVXPY solve runs on CPU through `jax.pure_callback`; this means the
    policy is *not* JIT-vmap-friendly. Use it inside a manual rollout loop
    (or accept the callback's host-roundtrip cost inside `lax.scan`).
    """

    horizon: int = 10
    control_cost: float = 1e-3
    mean_motion: float = 0.0  # injected by builder (depends on reference orbit)
    dt: float = 10.0
    dv_max: float = 1.0  # max delta-v per step (m/s)
    n_vehicles: int = 0  # populated by env
    command_cls: Any = None  # populated by env
    n_opp: int = 1  # number of opponents (guards), needed to slice obs

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
                "MPCBanditPolicy was called before the env injected `command_cls`. "
                "Use it via OrbitalGameEnv.rollout / SingleAgentView, or pass "
                "command_cls and n_vehicles explicitly."
            )

        n_b = self.n_vehicles
        d = 4  # HCW_RT state dim
        n_total = n_b + self.n_opp

        # FullObservation flatten: per-observer broadcast of (n_total, d), so
        # the flat layout is (n_b, n_total, d) reshaped to (n_b * n_total * d,).
        # Bandit i's own state is at observer-i, slot-i (own block comes first).
        obs_3d = agent_view.reshape((n_b, n_total, d))
        own_states = obs_3d[jnp.arange(n_b), jnp.arange(n_b), :]  # (n_b, d)

        # STM (A) and control (B) matrices; uppercase by convention.
        A, B = _build_hcw_rt_matrices(self.mean_motion, self.dt)  # noqa: N806
        A_j = jnp.asarray(A)  # noqa: N806
        B_j = jnp.asarray(B)  # noqa: N806

        def _host_solve(own_np, A_np, B_np):  # noqa: N803 — matrix-valued args
            return _solve_batch(
                np.asarray(own_np),
                np.asarray(A_np),
                np.asarray(B_np),
                self.horizon,
                self.dv_max,
                self.control_cost,
            ).astype(np.float32)

        cmd_template = self.command_cls.zeros(n_b)
        dv_dim = cmd_template.dv.shape[-1]  # 2 for RT, 3 for RTN/ECI
        result_shape = jax.ShapeDtypeStruct((n_b, 2), jnp.float32)
        dv_2d = jax.pure_callback(_host_solve, result_shape, own_states, A_j, B_j)

        if dv_dim == 2:
            dv_out = dv_2d
        else:
            # 3D action frame: pad cross-track with 0 (RT-only QP, by design).
            dv_out = jnp.concatenate([dv_2d, jnp.zeros((n_b, dv_dim - 2), dv_2d.dtype)], axis=-1)
        return cmd_template.replace(dv=dv_out), policy_state
