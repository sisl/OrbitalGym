"""Leaf values for search on LBG, in the units of :class:`LbgZeroSumReward`.

A search whose horizon is shorter than an intercept never reaches a catch or
a breach, so the leaf value has to carry the outcome. The leaf is a
terminal-event estimate plus the shaping potential::

    guard:   terminal_estimate(s) + shaping_gain * Phi(s)
    bandit: -terminal_estimate(s) - shaping_gain * Phi(s)

with ``Phi`` from :func:`orbitalgym.rewards.lbg_zero_sum.lbg_potential` on the
reward's own ``shaping_scale_m`` and ``home_weight``. The bandit's potential
is ``-Phi``, so each side adds its own, and the two leaves stay exact mirrors.

The plus sign is the value-initialization equivalence (Wiewiora 2003):
shaping a reward with ``Phi`` and initializing the value function with ``Phi``
produce the same greedy behavior, so a leaf that already carries ``Phi``
plays the role the shaping would have played beyond the horizon. Ranking a
depth-``D`` guard path by shaped rewards plus this leaf gives

    terminal_estimate(leaf) + (1 + g^D) Phi(leaf) - Phi(root)

up to the unshaped rewards collected on the way: the potential at the leaf
survives, and ``Phi(root)`` is a constant across the actions being compared.
Subtracting ``Phi`` instead would cancel it against the shaping that
telescoped along the path, leaving the search with the terminal estimate
alone and no guidance wherever that estimate is flat.

The reward weights and radii are required arguments, because a leaf value
built on weights the reward does not use orders states by a different game.
:func:`guard_leaf_value_from_game` and :func:`bandit_leaf_value_from_game`
read them off a scenario's ``cfg.reward_fn`` so the two cannot drift apart.

Time to an event is estimated in macro steps by a straight-line closing
approximation::

    t_hit = max(d - radius, 0) / (v_close * macro_dt)

``v_close`` is the average closing speed the mover can sustain over the
transfer, which the caller supplies. It is a speed, not the per-step delta-v
cap: a cap divided by ``dt`` is an acceleration and would put every event
hundreds of macro steps away, flattening the estimate. The straight-line model
ignores orbital curvature, the opponent's evasion, and the burn geometry, so
it is optimistic for a mover that must first null a relative velocity and
pessimistic for one already closing fast.

With ``g`` the per-macro-step discount and ``N`` the remaining macro steps,
the guard's terminal estimate is::

    terminal_estimate = r_catch * w(t_catch) - r_breach * w(t_breach)

``w`` is the weight the terminal bonus carries at an event ``t_hit`` macro
steps away. For ``g < 1`` it is the discount factor ``g^t_hit``. For
``g == 1`` discounting cannot express distance, so ``w`` is the linear
time-to-go weight of an undiscounted finite-horizon reach estimate::

    w(t_hit) = max(0, 1 - t_hit / N)

which pays the bonus in full at the radius, falls off linearly as the event
recedes, and pays nothing for an event beyond the horizon.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from orbitalgym.games.proximity import positions
from orbitalgym.rewards.lbg_zero_sum import LbgZeroSumReward, lbg_potential


def _distances(state: Any) -> tuple[jax.Array, jax.Array]:
    """Nearest guard-bandit and bandit-lady distances at the step endpoints."""
    guards = positions(state.guards)
    bandits = positions(state.bandits)
    d_gb = jnp.min(jnp.linalg.norm(guards[:, None, :] - bandits[None, :, :], axis=-1))
    d_bl = jnp.min(jnp.linalg.norm(bandits, axis=-1))
    return d_gb, d_bl


def _time_to_radius(
    d: jax.Array,
    radius_m: float,
    v_close_mps: float,
    macro_dt: float,
) -> jax.Array:
    """Macro steps needed to close ``d`` down to ``radius_m`` at ``v_close_mps``."""
    return jnp.maximum(d - radius_m, 0.0) / (v_close_mps * macro_dt)


def _event_weight(t_hit: jax.Array, discount: float, n_remaining: int) -> jax.Array:
    """Weight a terminal bonus carries when the event is ``t_hit`` macro steps away.

    ``g^t_hit`` while discounting, and the linear time-to-go weight
    ``max(0, 1 - t_hit / N)`` at ``g == 1``, where a discount factor would
    be 1 everywhere and the estimate would lose its distance dependence.
    """
    if discount == 1.0:
        return jnp.maximum(1.0 - t_hit / n_remaining, 0.0)
    return discount**t_hit


def _leaf_value(
    adapter: Any,
    sign: float,
    *,
    shaping_gain: float,
    shaping_scale_m: float,
    home_weight: float,
    r_catch: float,
    r_breach: float,
    catch_radius_m: float,
    breach_radius_m: float,
    v_close_guard_mps: float,
    v_close_bandit_mps: float,
    discount: float | None,
    n_remaining: int,
) -> Callable[[jax.Array], jax.Array]:
    """``sign * (terminal_estimate + shaping_gain * Phi)``, guard sign ``+1``."""
    g = adapter.discount() if discount is None else discount
    macro_dt = adapter.macro_dt

    def value(s_flat: jax.Array) -> jax.Array:
        state = adapter.unpack(s_flat)
        d_gb, d_bl = _distances(state)
        t_catch = _time_to_radius(d_gb, catch_radius_m, v_close_guard_mps, macro_dt)
        t_breach = _time_to_radius(d_bl, breach_radius_m, v_close_bandit_mps, macro_dt)
        terminal = r_catch * _event_weight(t_catch, g, n_remaining) - r_breach * _event_weight(
            t_breach, g, n_remaining
        )
        phi = lbg_potential(state, shaping_scale_m, home_weight)
        return (sign * (terminal + shaping_gain * phi)).astype(s_flat.dtype)

    return value


def guard_leaf_value(
    adapter: Any,
    *,
    shaping_gain: float,
    shaping_scale_m: float,
    home_weight: float,
    r_catch: float,
    r_breach: float,
    catch_radius_m: float,
    breach_radius_m: float,
    v_close_guard_mps: float,
    v_close_bandit_mps: float,
    discount: float | None = None,
    n_remaining: int = 32,
) -> Callable[[jax.Array], jax.Array]:
    """Guard-side value estimate in reward units.

    ``r_catch`` weighted by the estimated macro steps to a catch, less
    ``r_breach`` weighted by the estimated macro steps to a breach, plus
    ``shaping_gain`` times the potential at the leaf. ``discount`` defaults to
    ``adapter.discount()``, the per-macro-step discount.

    The reward weights and radii are required: an estimate built on
    weights the reward does not use orders states by a game nobody is
    playing. :func:`guard_leaf_value_from_game` reads them off a scenario.
    """
    return _leaf_value(
        adapter,
        1.0,
        shaping_gain=shaping_gain,
        shaping_scale_m=shaping_scale_m,
        home_weight=home_weight,
        r_catch=r_catch,
        r_breach=r_breach,
        catch_radius_m=catch_radius_m,
        breach_radius_m=breach_radius_m,
        v_close_guard_mps=v_close_guard_mps,
        v_close_bandit_mps=v_close_bandit_mps,
        discount=discount,
        n_remaining=n_remaining,
    )


def bandit_leaf_value(
    adapter: Any,
    *,
    shaping_gain: float,
    shaping_scale_m: float,
    home_weight: float,
    r_catch: float,
    r_breach: float,
    catch_radius_m: float,
    breach_radius_m: float,
    v_close_guard_mps: float,
    v_close_bandit_mps: float,
    discount: float | None = None,
    n_remaining: int = 32,
) -> Callable[[jax.Array], jax.Array]:
    """Bandit-side value estimate in reward units.

    The negation of :func:`guard_leaf_value`, matching the zero-sum geometry
    terms of the reward. The reward weights and radii are required;
    :func:`bandit_leaf_value_from_game` reads them off a scenario.
    """
    return _leaf_value(
        adapter,
        -1.0,
        shaping_gain=shaping_gain,
        shaping_scale_m=shaping_scale_m,
        home_weight=home_weight,
        r_catch=r_catch,
        r_breach=r_breach,
        catch_radius_m=catch_radius_m,
        breach_radius_m=breach_radius_m,
        v_close_guard_mps=v_close_guard_mps,
        v_close_bandit_mps=v_close_bandit_mps,
        discount=discount,
        n_remaining=n_remaining,
    )


def _reward_terms(cfg: Any) -> dict[str, float]:
    """The reward weights and radii a leaf value has to agree with.

    Reads them off ``cfg.reward_fn``, which must be an
    :class:`~orbitalgym.rewards.lbg_zero_sum.LbgZeroSumReward`: the leaf
    values are written in that reward's units and no other reward exposes
    the same terms.
    """
    reward_fn = getattr(cfg, "reward_fn", None)
    if not isinstance(reward_fn, LbgZeroSumReward):
        raise TypeError(
            "leaf values are expressed in the units of LbgZeroSumReward, so "
            "cfg.reward_fn must be one; got "
            f"{type(reward_fn).__name__}. Call guard_leaf_value or "
            "bandit_leaf_value with explicit weights and radii instead."
        )
    return {
        "shaping_gain": reward_fn.shaping_gain,
        "shaping_scale_m": reward_fn.shaping_scale_m,
        "home_weight": reward_fn.home_weight,
        "r_catch": reward_fn.r_catch,
        "r_breach": reward_fn.r_breach,
        "catch_radius_m": reward_fn.catch_radius_m,
        "breach_radius_m": reward_fn.breach_radius_m,
    }


def guard_leaf_value_from_game(
    adapter: Any,
    cfg: Any,
    *,
    v_close_guard_mps: float,
    v_close_bandit_mps: float,
    discount: float | None = None,
    n_remaining: int = 32,
) -> Callable[[jax.Array], jax.Array]:
    """:func:`guard_leaf_value` with the weights and radii read off ``cfg``."""
    return guard_leaf_value(
        adapter,
        **_reward_terms(cfg),
        v_close_guard_mps=v_close_guard_mps,
        v_close_bandit_mps=v_close_bandit_mps,
        discount=discount,
        n_remaining=n_remaining,
    )


def bandit_leaf_value_from_game(
    adapter: Any,
    cfg: Any,
    *,
    v_close_guard_mps: float,
    v_close_bandit_mps: float,
    discount: float | None = None,
    n_remaining: int = 32,
) -> Callable[[jax.Array], jax.Array]:
    """:func:`bandit_leaf_value` with the weights and radii read off ``cfg``."""
    return bandit_leaf_value(
        adapter,
        **_reward_terms(cfg),
        v_close_guard_mps=v_close_guard_mps,
        v_close_bandit_mps=v_close_bandit_mps,
        discount=discount,
        n_remaining=n_remaining,
    )
