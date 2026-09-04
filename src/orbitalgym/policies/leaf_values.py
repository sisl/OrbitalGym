"""Leaf values for search on LBG, in the units of :class:`LbgZeroSumReward`.

A search whose horizon is shorter than an intercept never reaches a catch or
a breach, so the leaf value has to carry the outcome. These estimates are on
the reward scale: they predict the shaping the side accrues over the
remaining macro steps plus the discounted terminal events, using the same
``alpha`` / ``r_catch`` / ``r_breach`` / radii the reward function uses.

Time to an event is estimated in macro steps by a straight-line closing
approximation::

    t_hit = max(d - radius, 0) / (v_close * macro_dt)

``v_close`` is the mover's acceleration-limited average closing speed, which
the caller supplies: its per-step delta-v cap divided by ``dt``. This ignores
orbital curvature, the opponent's evasion, and the burn geometry, so it is
optimistic for a mover that must first null a relative velocity and
pessimistic for one already closing fast. It is a value *estimate*; the
ordering it induces over states is what the search consumes.

With ``g`` the per-macro-step discount, ``N`` the remaining macro steps, and
``d`` the shaped distance, the guard's estimate is::

    V = -alpha * d_gb * (1 - g^N) / (1 - g) + r_catch * g^t_catch
        - r_breach * g^t_breach

and the bandit's is the mirror image::

    V = -alpha * d_bl * (1 - g^N) / (1 - g) + r_breach * g^t_breach
        - r_catch * g^t_catch

The geometric factor becomes the arithmetic sum ``N`` when ``g == 1``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def _positions(side_state: Any) -> jax.Array:
    if hasattr(side_state, "rtn"):
        return side_state.rtn[:, :3]
    return side_state.rt[:, :2]


def _distances(adapter: Any, s_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
    state = adapter.unpack(s_flat)
    guards = _positions(state.guards)
    bandits = _positions(state.bandits)
    d_gb = jnp.min(jnp.linalg.norm(guards[:, None, :] - bandits[None, :, :], axis=-1))
    d_bl = jnp.min(jnp.linalg.norm(bandits, axis=-1))
    return d_gb, d_bl


def _shaping_horizon(discount: float, n_remaining: int) -> float:
    """Sum of ``g^i`` over the remaining macro steps; arithmetic when ``g == 1``."""
    if discount == 1.0:
        return float(n_remaining)
    return (1.0 - discount**n_remaining) / (1.0 - discount)


def _time_to_radius(
    d: jax.Array,
    radius_m: float,
    v_close_mps: float,
    macro_dt: float,
) -> jax.Array:
    """Macro steps needed to close ``d`` down to ``radius_m`` at ``v_close_mps``."""
    return jnp.maximum(d - radius_m, 0.0) / (v_close_mps * macro_dt)


def guard_leaf_value(
    adapter: Any,
    *,
    alpha: float = 1e-3,
    r_catch: float = 1000.0,
    r_breach: float = 1000.0,
    catch_radius_m: float = 50.0,
    breach_radius_m: float = 5.0,
    v_close_guard_mps: float,
    v_close_bandit_mps: float,
    discount: float | None = None,
    n_remaining: int = 32,
) -> Callable[[jax.Array], jax.Array]:
    """Guard-side value estimate in reward units.

    Sums the remaining guard-bandit shaping, ``r_catch`` discounted by the
    estimated macro steps to a catch, and ``-r_breach`` discounted by the
    estimated macro steps to a breach. ``discount`` defaults to
    ``adapter.discount()``, the per-macro-step discount.
    """
    g = adapter.discount() if discount is None else discount
    horizon = _shaping_horizon(g, n_remaining)
    macro_dt = adapter.macro_dt

    def value(s_flat: jax.Array) -> jax.Array:
        d_gb, d_bl = _distances(adapter, s_flat)
        t_catch = _time_to_radius(d_gb, catch_radius_m, v_close_guard_mps, macro_dt)
        t_breach = _time_to_radius(d_bl, breach_radius_m, v_close_bandit_mps, macro_dt)
        v = -alpha * d_gb * horizon + r_catch * g**t_catch - r_breach * g**t_breach
        return v.astype(s_flat.dtype)

    return value


def bandit_leaf_value(
    adapter: Any,
    *,
    alpha: float = 1e-3,
    r_catch: float = 1000.0,
    r_breach: float = 1000.0,
    catch_radius_m: float = 50.0,
    breach_radius_m: float = 5.0,
    v_close_guard_mps: float,
    v_close_bandit_mps: float,
    discount: float | None = None,
    n_remaining: int = 32,
) -> Callable[[jax.Array], jax.Array]:
    """Bandit-side value estimate in reward units.

    The mirror of :func:`guard_leaf_value`: remaining bandit-lady shaping,
    ``r_breach`` discounted by the estimated macro steps to a breach, and
    ``-r_catch`` discounted by the estimated macro steps to a catch.
    """
    g = adapter.discount() if discount is None else discount
    horizon = _shaping_horizon(g, n_remaining)
    macro_dt = adapter.macro_dt

    def value(s_flat: jax.Array) -> jax.Array:
        d_gb, d_bl = _distances(adapter, s_flat)
        t_catch = _time_to_radius(d_gb, catch_radius_m, v_close_guard_mps, macro_dt)
        t_breach = _time_to_radius(d_bl, breach_radius_m, v_close_bandit_mps, macro_dt)
        v = -alpha * d_bl * horizon + r_breach * g**t_breach - r_catch * g**t_catch
        return v.astype(s_flat.dtype)

    return value
