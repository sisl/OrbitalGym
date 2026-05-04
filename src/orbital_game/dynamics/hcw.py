"""Hill-Clohessy-Wiltshire (HCW) relative-motion dynamics.

Ships two closed-form STM steps:
  - hcw_rt_step:  2D (radial + along-track) — 4-dim state per vehicle
  - hcw_rtn_step: 3D (radial + along-track + cross-track) — 6-dim state per vehicle

The state leaf is raw jax.Array of shape (n, 4) or (n, 6); the component wrapper
(RTState / RTNState) just names it. These functions take and return the raw array
so they compose naturally with vmap over vehicles.

`applied.dv` is treated as a velocity impulse applied at the START of the step;
this convention keeps impulsive and continuous actuators interchangeable under the
same dynamics signature.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbital_game.registry import DynamicsKey, DynamicsKind, Frame, register


def _hcw_in_plane_stm(s: jax.Array, c: jax.Array, n: float, dt: float) -> jax.Array:
    """Closed-form 4x4 HCW in-plane STM (Vallado).

    Shared between hcw_rt_step (2D) and hcw_rtn_step (3D, in-plane block).
    s, c are sin/cos of (n * dt); n is mean motion (rad/s); dt is the step (seconds).

    Matrix layout:
      [x ]   [ 4 - 3c,         0,  s/n,        2(1-c)/n ] [x0  ]
      [y ]   [ 6(s - n dt),    1, -2(1-c)/n,  (4s - 3 n dt)/n ] [y0  ]
      [xd]   [ 3 n s,          0,  c,          2 s      ] [xd0 ]
      [yd]   [-6 n (1 - c),    0, -2 s,        4 c - 3  ] [yd0 ]
    """
    return jnp.array(
        [
            [4 - 3 * c, 0.0, s / n, 2 * (1 - c) / n],
            [6 * (s - n * dt), 1.0, -2 * (1 - c) / n, (4 * s - 3 * n * dt) / n],
            [3 * n * s, 0.0, c, 2 * s],
            [-6 * n * (1 - c), 0.0, -2 * s, 4 * c - 3],
        ]
    )


@register(DynamicsKey.HCW_RT, frame=Frame.RT, kind=DynamicsKind.RELATIVE)
def hcw_rt_step(state: jax.Array, dv: jax.Array, params, dt: float) -> jax.Array:
    """In-plane HCW step via closed-form 4x4 STM.

    state: (n, 4) — (x, y, xdot, ydot) per vehicle. x=radial, y=along-track.
    dv:    (n, 2) — (dvx, dvy) velocity impulse applied at start of interval.
    params: object with float attr mean_motion (rad/s).
    dt:    scalar seconds.
    """
    n = params.mean_motion
    s = jnp.sin(n * dt)
    c = jnp.cos(n * dt)
    phi = _hcw_in_plane_stm(s, c, n, dt)

    # Apply impulse at start: add dv to velocity components.
    s0 = state.at[:, 2].add(dv[:, 0]).at[:, 3].add(dv[:, 1])
    # state is (n, 4) — batch-first JAX convention. x_new_row = x_old_row @ phi^T
    # is the row-vector equivalent of the textbook form phi @ x_col.
    return s0 @ phi.T


@register(DynamicsKey.HCW_RTN, frame=Frame.RTN, kind=DynamicsKind.RELATIVE)
def hcw_rtn_step(state: jax.Array, dv: jax.Array, params, dt: float) -> jax.Array:
    """Full 3D HCW step. In-plane (R, T) uses the same 4x4 STM as hcw_rt_step;
    cross-track (N) is a decoupled 2D harmonic oscillator with the same mean motion.

    state: (n, 6) — (R, T, N, Rdot, Tdot, Ndot) per vehicle.
    dv:    (n, 3) — (dvR, dvT, dvN) velocity impulse applied at start of interval.
    params: object with float attr mean_motion (rad/s).
    dt:    scalar seconds.
    """
    n = params.mean_motion
    s = jnp.sin(n * dt)
    c = jnp.cos(n * dt)

    # Add impulse to velocity components up front.
    s0 = state
    s0 = s0.at[:, 3].add(dv[:, 0])
    s0 = s0.at[:, 4].add(dv[:, 1])
    s0 = s0.at[:, 5].add(dv[:, 2])

    # Separate in-plane and cross-track substates.
    in_plane = jnp.stack([s0[:, 0], s0[:, 1], s0[:, 3], s0[:, 4]], axis=-1)  # (n, 4)
    out_plane = jnp.stack([s0[:, 2], s0[:, 5]], axis=-1)  # (n, 2)

    phi_in = _hcw_in_plane_stm(s, c, n, dt)
    # Cross-track STM: [[cos, sin/n], [-n sin, cos]]
    phi_out = jnp.array(
        [
            [c, s / n],
            [-n * s, c],
        ]
    )

    # Same batch-first convention as hcw_rt_step: each row is a vehicle.
    new_in = in_plane @ phi_in.T
    new_out = out_plane @ phi_out.T

    return jnp.stack(
        [
            new_in[:, 0],  # R
            new_in[:, 1],  # T
            new_out[:, 0],  # N
            new_in[:, 2],  # Rdot
            new_in[:, 3],  # Tdot
            new_out[:, 1],  # Ndot
        ],
        axis=-1,
    )
