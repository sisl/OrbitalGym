"""Feedback and non-reactive policies of both games, written on the relative state.

A pursuit law acts on the state of the pursuing vehicle relative to its target,
``x = [r; v]`` in the radial, along-track, cross-track frame. The PD and LQR
laws are OrbitalGym's; the direct, HCW intercept, random, and evasion laws are
specific to this study. ``limit_impulse`` applies the thrust limit and the
remaining delta-v budget of the constant-mass actuation model.
"""

import jax
import jax.numpy as jnp
import numpy as np

from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.policies.heuristic.feedback import hcw_lqr_gain, lqr_impulse, pd_impulse

PURSUERS = ("direct", "lqr", "hcw", "coast", "random", "pd")
EVADERS = ("coast", "flee", "transverse", "random")


def limit_impulse(command, cap, remaining):
    """Scale an impulse to the thrust limit and the remaining budget, keeping its direction."""
    norm = jnp.linalg.norm(command)
    scale = jnp.minimum(
        1.0, jnp.minimum(cap, jnp.maximum(remaining, 0.0)) / jnp.maximum(norm, 1e-30)
    )
    return command * scale


def hcw_intercept_matrix(mean_motion, lookahead):
    """Matrix ``M`` such that ``-M x`` brings the relative position to zero after ``lookahead``.

    The target is assumed to coast, and no terminal velocity is imposed.
    """
    phi = np.asarray(hcw_rtn_stm(mean_motion, lookahead))
    if np.linalg.cond(phi[:3, 3:]) > 1e10:
        raise ValueError("HCW intercept lookahead is numerically singular")
    return jnp.asarray(np.linalg.solve(phi[:3, 3:], phi[:3]))


def controller_matrices(mean_motion, dt, lookahead):
    """The LQR gain for one step ``dt`` and the HCW intercept matrix for ``lookahead``."""
    return hcw_lqr_gain(mean_motion, dt), hcw_intercept_matrix(mean_motion, lookahead)


def random_impulse(key, cap):
    """Uniform in the ball of admissible impulses; independent at each step."""
    direction_key, radius_key = jax.random.split(key)
    direction = jax.random.normal(direction_key, (3,))
    direction /= jnp.maximum(jnp.linalg.norm(direction), 1e-30)
    return cap * jax.random.uniform(radius_key) ** (1.0 / 3.0) * direction


def pursuit_command(index, relative, matrices, cap, key, dt):
    """Unlimited impulse of pursuit law ``PURSUERS[index]`` on the pursuer-minus-target state."""
    gain, intercept = matrices
    return jax.lax.switch(
        index,
        (
            lambda x: -cap * x[:3] / jnp.maximum(jnp.linalg.norm(x[:3]), 1e-30),
            lambda x: lqr_impulse(gain, x),
            lambda x: -intercept @ x,
            lambda x: jnp.zeros(3),
            lambda x: random_impulse(key, cap),
            lambda x: pd_impulse(x, dt),
        ),
        relative,
    )


def evasion_command(index, relative, cap, key):
    """Impulse of evasion law ``EVADERS[index]`` on the pursuer-minus-evader state.

    The transverse law thrusts along ``away x e_N``, perpendicular to the line
    of sight and to the cross-track axis; the radial axis replaces ``e_N`` when
    the line of sight is within 25.8 degrees of cross-track.
    """
    away = -relative[:3] / jnp.maximum(jnp.linalg.norm(relative[:3]), 1e-30)
    axis = jnp.where(jnp.abs(away[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0]))
    transverse = jnp.cross(away, axis)
    transverse /= jnp.maximum(jnp.linalg.norm(transverse), 1e-30)
    return jax.lax.switch(
        index,
        (
            lambda: jnp.zeros(3),
            lambda: cap * away,
            lambda: cap * transverse,
            lambda: random_impulse(key, cap),
        ),
    )
