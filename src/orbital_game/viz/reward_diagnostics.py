"""Game-aware reward-construction diagnostics.

Each plot overlays *the signal driving the reward* against the *actual reward
recorded in the rollout*, so you can confirm by eye that the reward function
implements its intended semantics. Useful as a sanity check whenever you wire
up a new game or change a reward definition.

Dispatch via :func:`plot_reward_diagnostic` — it picks the right plot based on
``cfg.game`` and falls back to a generic guard-vs-bandit reward plot for
games it doesn't recognize.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from orbital_game.env.types import Trajectory
from orbital_game.games.lady_bandit_guard import LadyBanditGuard
from orbital_game.games.pursuit_evasion import PursuitEvasion
from orbital_game.viz.per_rollout import plot_rollout_rewards


def _positions(side_state: Any) -> np.ndarray:
    if hasattr(side_state, "rtn"):
        return np.asarray(side_state.rtn[..., :3])
    if hasattr(side_state, "rt"):
        rt = np.asarray(side_state.rt)
        rn = np.zeros(rt.shape[:-1] + (1,), dtype=rt.dtype)
        return np.concatenate([rt[..., :2], rn], axis=-1)
    raise AttributeError("Side state has no `rtn` or `rt` field.")


def plot_pursuit_evasion_diagnostic(
    traj: Trajectory,
    capture_distance_m: float,
    fig: Any | None = None,
) -> Any:
    """Two-panel diagnostic: relative distance vs reward.

    Top: ``|r_guard - r_bandit|`` over time, with the capture-distance
    threshold drawn as a horizontal line.
    Bottom: guard and bandit rewards. For Pursuit-Evasion they should be
    equal-and-opposite (zero-sum).
    """
    g_xyz = _positions(traj.env_state.guards)  # (T, n_g, 3)
    b_xyz = _positions(traj.env_state.bandits)  # (T, n_b, 3)
    rel = g_xyz[:, 0, :] - b_xyz[:, 0, :]  # 1v1 by construction
    dist = np.linalg.norm(rel, axis=-1)  # (T,)

    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    ax_top = fig.add_subplot(2, 1, 1)
    ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)

    ax_top.plot(dist, color="black", linewidth=2.0, label="|r_guard - r_bandit|")
    ax_top.axhline(
        capture_distance_m,
        color="red",
        linestyle="--",
        label=f"capture threshold ({capture_distance_m:g} m)",
    )
    ax_top.set_ylabel("relative distance (m)")
    ax_top.grid(True)
    ax_top.legend(loc="best", fontsize=8)
    ax_top.set_title("Pursuit-Evasion reward diagnostic")

    plot_rollout_rewards(traj, ax=ax_bot)

    # Zero-sum check (per_side): annotate any drift from |g + b| ≈ 0.
    g_r = jnp.asarray(traj.sides.guard.reward)
    b_r = jnp.asarray(traj.sides.bandit.reward)
    if g_r.ndim == 1 and b_r.ndim == 1:
        max_abs_sum = float(jnp.max(jnp.abs(g_r + b_r)))
        ax_bot.text(
            0.02,
            0.95,
            f"max |g+b| = {max_abs_sum:.3g}  (zero-sum check)",
            transform=ax_bot.transAxes,
            fontsize=8,
            verticalalignment="top",
        )

    fig.tight_layout()
    return fig


def plot_lady_bandit_guard_diagnostic(
    traj: Trajectory,
    breach_distance_m: float,
    fig: Any | None = None,
) -> Any:
    """Two-panel diagnostic: guard distance(s) to reference vs guard reward.

    Top: per-guard distance to the reference-orbit origin (RTN frame) with
    the breach threshold drawn as a horizontal line. Episodes terminate
    when any guard crosses below this line.
    Bottom: guard reward — for the LBG default reward, ``reward(t)`` should
    match ``-sum_i |guard_i(t)|``.
    """
    g_xyz = _positions(traj.env_state.guards)  # (T, n_g, 3)
    distances = np.linalg.norm(g_xyz, axis=-1)  # (T, n_g)

    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    ax_top = fig.add_subplot(2, 1, 1)
    ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)

    n_g = distances.shape[1]
    cmap = mpl.colormaps["Blues"]
    for i in range(n_g):
        c = cmap(0.45 + 0.5 * i / max(n_g - 1, 1))
        ax_top.plot(distances[:, i], color=c, label=f"guard {i}")
    ax_top.axhline(
        breach_distance_m,
        color="red",
        linestyle="--",
        label=f"breach threshold ({breach_distance_m:g} m)",
    )
    ax_top.set_ylabel("|guard − ref| (m)")
    ax_top.grid(True)
    ax_top.legend(loc="best", fontsize=8)
    ax_top.set_title("Lady-Bandit-Guard reward diagnostic")

    g_reward = jnp.asarray(traj.sides.guard.reward)
    if g_reward.ndim == 1:
        ax_bot.plot(g_reward, color="tab:blue", linewidth=2.0, label="guard reward")
    elif g_reward.ndim == 2:
        for i in range(g_reward.shape[1]):
            c = cmap(0.45 + 0.5 * i / max(g_reward.shape[1] - 1, 1))
            ax_bot.plot(g_reward[:, i], color=c, label=f"guard {i}")
    ax_bot.set_xlabel("step")
    ax_bot.set_ylabel("reward")
    ax_bot.grid(True)
    ax_bot.legend(loc="best", fontsize=8)

    # Sanity-check overlay: for the default LBG reward, expected ≈ -sum_i |d_i|.
    expected = -np.sum(distances, axis=1)
    ax_bot.plot(
        expected,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="expected: -Σ|guard − ref|",
    )
    ax_bot.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


def plot_reward_diagnostic(
    traj: Trajectory,
    cfg: Any,
    fig: Any | None = None,
) -> Any:
    """Dispatch a reward-construction diagnostic based on ``cfg.game`` type.

    Falls back to a generic guard-vs-bandit reward plot if the game type
    isn't specifically handled.
    """
    game = getattr(cfg, "game", None)
    if isinstance(game, PursuitEvasion):
        return plot_pursuit_evasion_diagnostic(traj, game.capture_distance_m, fig=fig)
    if isinstance(game, LadyBanditGuard):
        return plot_lady_bandit_guard_diagnostic(traj, game.breach_distance_m, fig=fig)

    if fig is None:
        fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    plot_rollout_rewards(traj, ax=ax)
    ax.set_title(f"Reward diagnostic — game={type(game).__name__}")
    fig.tight_layout()
    return fig
