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

import jax
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


def plot_sun_blocking_diagnostic(
    traj: Trajectory,
    cfg: Any,
    fig: Any | None = None,
) -> Any:
    """Three-panel SB diagnostic: -û_BG·û_BS, |bandit−guard|, and rewards.

    Reconstructs the angular signal (negated dot product of unit vectors
    from bandit to guard and bandit to sun) and the bandit-guard distance
    from the recorded trajectory, alongside the bandit + guard rewards.
    With the KSP-DG-aligned reward this should produce a near-exact
    overlay of `(-dot) * exp(-decay*(d-d_target)²)` on top of the recorded
    bandit reward.
    """
    import astrojax

    from orbital_game.games._frames import vehicle_eci_position
    from orbital_game.games.sun_blocking import _epoch_from_mjd

    g_xyz = _positions(traj.env_state.guards)  # (T, n_g, 3)  RTN
    n_steps = g_xyz.shape[0]
    times = np.asarray(traj.env_state.t)  # (T,)

    # Recompute ECI positions per timestep — small T, this is fine in numpy/jax.
    guard_eci = np.zeros((n_steps, 3))
    bandit_eci = np.zeros((n_steps, 3))
    sun_eci = np.zeros((n_steps, 3))

    # Build a vehicle-state-shaped object per timestep for vehicle_eci_position.
    # We rely on traj.env_state.guards being indexable by [t, 0, ...].
    for i in range(n_steps):
        gv = jax.tree_util.tree_map(lambda x, i=i: x[i, 0], traj.env_state.guards)
        bv = jax.tree_util.tree_map(lambda x, i=i: x[i, 0], traj.env_state.bandits)
        guard_eci[i] = np.asarray(
            vehicle_eci_position(gv, cfg.reference_orbit, cfg.epoch_mjd_utc, times[i])
        )
        bandit_eci[i] = np.asarray(
            vehicle_eci_position(bv, cfg.reference_orbit, cfg.epoch_mjd_utc, times[i])
        )
        epoch = _epoch_from_mjd(cfg.epoch_mjd_utc + float(times[i]) / 86400.0)
        sun_eci[i] = np.asarray(astrojax.sun_position(epoch))

    # Angular factor and range factor.
    rel_bg = guard_eci - bandit_eci
    rel_bs = sun_eci - bandit_eci
    d_bg = np.linalg.norm(rel_bg, axis=1)
    u_bg = rel_bg / (d_bg[:, None] + 1e-12)
    u_bs = rel_bs / (np.linalg.norm(rel_bs, axis=1, keepdims=True) + 1e-12)
    angular = -np.einsum("td,td->t", u_bg, u_bs)
    d_target = cfg.game.target_viewing_distance_m
    decay = cfg.game.range_decay_coef
    range_factor = np.exp(-decay * (d_bg - d_target) ** 2)
    reconstructed = angular * range_factor

    if fig is None:
        fig = plt.figure(figsize=(10, 9))
    ax_a = fig.add_subplot(3, 1, 1)
    ax_d = fig.add_subplot(3, 1, 2, sharex=ax_a)
    ax_r = fig.add_subplot(3, 1, 3, sharex=ax_a)

    ax_a.plot(angular, color="tab:purple", linewidth=1.5)
    ax_a.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax_a.axhline(-1.0, color="black", linewidth=0.5, linestyle=":")
    ax_a.set_ylabel("-û_BG · û_BS")
    ax_a.grid(True)
    ax_a.set_title("Sun-Blocking reward diagnostic")

    ax_d.plot(d_bg, color="tab:cyan", linewidth=1.5, label="|bandit − guard|")
    ax_d.axhline(d_target, color="red", linestyle="--", label=f"d_target ({d_target:g} m)")
    ax_d.set_ylabel("range (m)")
    ax_d.grid(True)
    ax_d.legend(loc="best", fontsize=8)

    g_r = np.asarray(traj.sides.guard.reward)
    b_r = np.asarray(traj.sides.bandit.reward)
    ax_r.plot(b_r, color="tab:red", linewidth=2.0, label="bandit reward")
    ax_r.plot(g_r, color="tab:blue", linewidth=2.0, label="guard reward")
    ax_r.plot(
        reconstructed,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="reconstructed bandit reward",
    )
    ax_r.set_xlabel("step")
    ax_r.set_ylabel("reward")
    ax_r.grid(True)
    ax_r.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


def plot_observation_blocking_diagnostic(
    traj: Trajectory,
    cfg: Any,
    fig: Any | None = None,
) -> Any:
    """Four-panel OB diagnostic: visibility, angular, range, rewards.

    The visibility panel shades intervals where the target is below
    `min_elevation_deg` from the guard — those are the windows where the
    reward is zero by gate, regardless of geometry.
    """
    import astrojax

    from orbital_game.games._frames import vehicle_eci_position
    from orbital_game.games.observation_blocking import (
        _epoch_from_mjd,
        target_visible_from_guard,
    )

    g_xyz = _positions(traj.env_state.guards)
    n_steps = g_xyz.shape[0]
    times = np.asarray(traj.env_state.t)

    guard_eci = np.zeros((n_steps, 3))
    bandit_eci = np.zeros((n_steps, 3))
    target_eci = np.zeros((n_steps, 3))
    visible = np.zeros(n_steps, dtype=bool)

    for i in range(n_steps):
        gv = jax.tree_util.tree_map(lambda x, i=i: x[i, 0], traj.env_state.guards)
        bv = jax.tree_util.tree_map(lambda x, i=i: x[i, 0], traj.env_state.bandits)
        guard_eci[i] = np.asarray(
            vehicle_eci_position(gv, cfg.reference_orbit, cfg.epoch_mjd_utc, times[i])
        )
        bandit_eci[i] = np.asarray(
            vehicle_eci_position(bv, cfg.reference_orbit, cfg.epoch_mjd_utc, times[i])
        )
        epoch = _epoch_from_mjd(cfg.epoch_mjd_utc + float(times[i]) / 86400.0)
        eop = astrojax.zero_eop()
        rot = astrojax.rotation_ecef_to_eci(eop, epoch)
        target_eci[i] = np.asarray(rot @ cfg.game.target_ecef_m)
        visible[i] = bool(
            target_visible_from_guard(
                jnp.asarray(target_eci[i]), jnp.asarray(guard_eci[i]), cfg.game.min_elevation_deg
            )
        )

    rel_bg = guard_eci - bandit_eci
    rel_bt = target_eci - bandit_eci
    d_bg = np.linalg.norm(rel_bg, axis=1)
    u_bg = rel_bg / (d_bg[:, None] + 1e-12)
    u_bt = rel_bt / (np.linalg.norm(rel_bt, axis=1, keepdims=True) + 1e-12)
    angular = -np.einsum("td,td->t", u_bg, u_bt)

    # Elevation in degrees for the visibility plot.
    up = target_eci / (np.linalg.norm(target_eci, axis=1, keepdims=True) + 1e-12)
    dir_to_guard = (guard_eci - target_eci) / (
        np.linalg.norm(guard_eci - target_eci, axis=1, keepdims=True) + 1e-12
    )
    sin_el = np.einsum("td,td->t", up, dir_to_guard)
    elevation_deg = np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

    if fig is None:
        fig = plt.figure(figsize=(10, 11))
    ax_v = fig.add_subplot(4, 1, 1)
    ax_a = fig.add_subplot(4, 1, 2, sharex=ax_v)
    ax_d = fig.add_subplot(4, 1, 3, sharex=ax_v)
    ax_r = fig.add_subplot(4, 1, 4, sharex=ax_v)

    ax_v.plot(elevation_deg, color="tab:green", linewidth=1.5)
    ax_v.axhline(
        cfg.game.min_elevation_deg,
        color="red",
        linestyle="--",
        label=f"min_elevation_deg ({cfg.game.min_elevation_deg:g}°)",
    )
    # Shade invisible intervals.
    invis_mask = (~visible).tolist()
    ax_v.fill_between(
        np.arange(n_steps),
        -90,
        90,
        where=invis_mask,
        alpha=0.1,
        color="grey",
        step="mid",
        label="target not visible",
    )
    ax_v.set_ylabel("elevation (°)")
    ax_v.grid(True)
    ax_v.legend(loc="best", fontsize=8)
    ax_v.set_title("Observation-Blocking reward diagnostic")

    ax_a.plot(angular, color="tab:purple", linewidth=1.5)
    ax_a.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax_a.axhline(-1.0, color="black", linewidth=0.5, linestyle=":")
    ax_a.set_ylabel("-û_BG · û_BT")
    ax_a.grid(True)

    d_target = cfg.game.target_viewing_distance_m
    ax_d.plot(d_bg, color="tab:cyan", linewidth=1.5, label="|bandit − guard|")
    ax_d.axhline(d_target, color="red", linestyle="--", label=f"d_target ({d_target:g} m)")
    ax_d.set_ylabel("range (m)")
    ax_d.grid(True)
    ax_d.legend(loc="best", fontsize=8)

    g_r = np.asarray(traj.sides.guard.reward)
    b_r = np.asarray(traj.sides.bandit.reward)
    ax_r.plot(b_r, color="tab:red", linewidth=2.0, label="bandit reward")
    ax_r.plot(g_r, color="tab:blue", linewidth=2.0, label="guard reward")
    ax_r.set_xlabel("step")
    ax_r.set_ylabel("reward")
    ax_r.grid(True)
    ax_r.legend(loc="best", fontsize=8)

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
    from orbital_game.games.observation_blocking import ObservationBlocking
    from orbital_game.games.sun_blocking import SunBlocking

    game = getattr(cfg, "game", None)
    if isinstance(game, PursuitEvasion):
        return plot_pursuit_evasion_diagnostic(traj, game.capture_distance_m, fig=fig)
    if isinstance(game, LadyBanditGuard):
        return plot_lady_bandit_guard_diagnostic(traj, game.breach_distance_m, fig=fig)
    if isinstance(game, SunBlocking):
        return plot_sun_blocking_diagnostic(traj, cfg, fig=fig)
    if isinstance(game, ObservationBlocking):
        return plot_observation_blocking_diagnostic(traj, cfg, fig=fig)

    if fig is None:
        fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    plot_rollout_rewards(traj, ax=ax)
    ax.set_title(f"Reward diagnostic — game={type(game).__name__}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Position-sweep diagnostics — render the reward as a 2D surface over a grid
# of one side's positions, with everything else fixed. Validates the actual
# kernel implementation (called via jax.vmap), not a re-derivation.
# ---------------------------------------------------------------------------


_AXIS_PAIR_INDICES = {"rt": (0, 1), "rn": (0, 2), "tn": (1, 2)}


def _sweep_grid(grid_extent_m: float, grid_steps: int, axis_pair: str):
    """Build the 2D RTN-offset grid for a position sweep.

    Returns (xs, ys, rtn_offsets) where rtn_offsets has shape (G, G, 3) with
    the un-swept axis = 0 and the swept pair set from a meshgrid.
    """
    if axis_pair not in _AXIS_PAIR_INDICES:
        raise ValueError(f"axis_pair must be one of {list(_AXIS_PAIR_INDICES)}; got {axis_pair!r}")
    a, b = _AXIS_PAIR_INDICES[axis_pair]
    grid = jnp.linspace(-grid_extent_m, grid_extent_m, grid_steps)
    xs, ys = jnp.meshgrid(grid, grid, indexing="ij")
    rtn = jnp.zeros((grid_steps, grid_steps, 3))
    rtn = rtn.at[..., a].set(xs)
    rtn = rtn.at[..., b].set(ys)
    return xs, ys, rtn


def _render_sweep_surface(xs, ys, zs, axis_pair: str, title: str, vmin, vmax, cmap, fig):
    """Common matplotlib plumbing for the four sweep functions."""
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    xs_n = np.asarray(xs)
    ys_n = np.asarray(ys)
    zs_n = np.asarray(zs)
    surf = ax.plot_surface(
        xs_n,
        ys_n,
        zs_n,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        linewidth=0,
        antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, label="reward [-]")
    ax.set_xlabel(f"{axis_pair[0].upper()}-pos [m]")
    ax.set_ylabel(f"{axis_pair[1].upper()}-pos [m]")
    ax.set_zlabel("reward [-]")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_sun_blocking_position_sweep(
    cfg: Any,
    *,
    grid_extent_m: float = 2000.0,
    grid_steps: int = 64,
    axis_pair: str = "rt",
    t_seconds: float = 0.0,
    fig: Any | None = None,
) -> Any:
    """Sweep bandit position over a 2D RTN-pair plane and render reward surface.

    Guard is fixed at the reference-orbit origin (RTN=0). Sun is fixed at the
    ECI position corresponding to ``cfg.epoch_mjd_utc + t_seconds``. The
    actual ``sun_blocking_kernel`` is called via jax.vmap on a grid of bandit
    positions (converted RTN→ECI per cell using the same frame helpers as
    the reward function).
    """
    import astrojax

    from orbital_game.games._frames import reference_orbit_eci_at_t, rtn_basis
    from orbital_game.games.sun_blocking import _epoch_from_mjd, sun_blocking_kernel

    ref_pos, ref_vel = reference_orbit_eci_at_t(cfg.reference_orbit, cfg.epoch_mjd_utc, t_seconds)
    rot = rtn_basis(ref_pos, ref_vel)
    guard_eci = ref_pos
    epoch = _epoch_from_mjd(cfg.epoch_mjd_utc + t_seconds / 86400.0)
    sun_eci = astrojax.sun_position(epoch)

    xs, ys, rtn = _sweep_grid(grid_extent_m, grid_steps, axis_pair)
    rtn_flat = rtn.reshape(-1, 3)
    bandit_eci_flat = ref_pos[None, :] + rtn_flat @ rot.T

    kernel = jax.vmap(
        lambda b: sun_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=b,
            sun_eci=sun_eci,
            target_viewing_distance_m=cfg.game.target_viewing_distance_m,
            range_decay_coef=cfg.game.range_decay_coef,
        )
    )
    rewards = kernel(bandit_eci_flat).reshape(grid_steps, grid_steps)
    title = (
        f"Sun-Blocking reward surface — bandit position "
        f"({axis_pair.upper()} plane, t={t_seconds:g}s)"
    )
    return _render_sweep_surface(
        xs,
        ys,
        rewards,
        axis_pair,
        title=title,
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
        fig=fig,
    )


def plot_observation_blocking_position_sweep(
    cfg: Any,
    *,
    grid_extent_m: float = 2000.0,
    grid_steps: int = 64,
    axis_pair: str = "rt",
    t_seconds: float = 0.0,
    fig: Any | None = None,
) -> Any:
    """Sweep bandit position over a 2D RTN-pair plane (OB).

    Same as the SB sweep but with the surface target replacing the sun and
    the visibility gate applied. The function prints `visible=True/False`
    for the chosen ``t_seconds``; if the target is not visible from the
    guard at that instant, the surface is uniformly zero.
    """
    import astrojax

    from orbital_game.games._frames import reference_orbit_eci_at_t, rtn_basis
    from orbital_game.games.observation_blocking import (
        _epoch_from_mjd,
        observation_blocking_kernel,
        target_visible_from_guard,
    )

    ref_pos, ref_vel = reference_orbit_eci_at_t(cfg.reference_orbit, cfg.epoch_mjd_utc, t_seconds)
    rot = rtn_basis(ref_pos, ref_vel)
    guard_eci = ref_pos
    epoch = _epoch_from_mjd(cfg.epoch_mjd_utc + t_seconds / 86400.0)
    eop = astrojax.zero_eop()
    rot_ecef_to_eci = astrojax.rotation_ecef_to_eci(eop, epoch)
    target_eci = rot_ecef_to_eci @ cfg.game.target_ecef_m

    visible = bool(target_visible_from_guard(target_eci, guard_eci, cfg.game.min_elevation_deg))
    print(f"plot_observation_blocking_position_sweep: visible={visible} at t={t_seconds:g}s")

    xs, ys, rtn = _sweep_grid(grid_extent_m, grid_steps, axis_pair)
    rtn_flat = rtn.reshape(-1, 3)
    bandit_eci_flat = ref_pos[None, :] + rtn_flat @ rot.T

    kernel = jax.vmap(
        lambda b: observation_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=b,
            target_eci=target_eci,
            target_viewing_distance_m=cfg.game.target_viewing_distance_m,
            range_decay_coef=cfg.game.range_decay_coef,
            min_elevation_deg=cfg.game.min_elevation_deg,
        )
    )
    rewards = kernel(bandit_eci_flat).reshape(grid_steps, grid_steps)
    title = (
        f"Observation-Blocking reward surface — bandit position "
        f"({axis_pair.upper()} plane, t={t_seconds:g}s, visible={visible})"
    )
    return _render_sweep_surface(
        xs,
        ys,
        rewards,
        axis_pair,
        title=title,
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
        fig=fig,
    )


def plot_pursuit_evasion_position_sweep(
    cfg: Any,
    *,
    grid_extent_m: float = 2000.0,
    grid_steps: int = 64,
    axis_pair: str = "rt",
    fig: Any | None = None,
) -> Any:
    """Sweep bandit position over a 2D RTN-pair plane (PE).

    Reward = -|guard − bandit| for the bandit. Surface is an inverted cone
    centered at the guard (origin in RTN). Included for symmetry with the
    SB/OB notebooks; for PE the shape is well-known.
    """
    xs, ys, rtn = _sweep_grid(grid_extent_m, grid_steps, axis_pair)
    distances = jnp.linalg.norm(rtn, axis=-1)
    rewards = -distances
    vmin = float(rewards.min())
    return _render_sweep_surface(
        xs,
        ys,
        rewards,
        axis_pair,
        title=f"Pursuit-Evasion bandit reward surface ({axis_pair.upper()} plane)",
        vmin=vmin,
        vmax=0.0,
        cmap="viridis",
        fig=fig,
    )


def plot_lady_bandit_guard_position_sweep(
    cfg: Any,
    *,
    grid_extent_m: float = 2000.0,
    grid_steps: int = 64,
    axis_pair: str = "rt",
    fig: Any | None = None,
) -> Any:
    """Sweep guard position over a 2D RTN-pair plane (LBG).

    LBG reward is `-Σ|guard|` — the bandit doesn't enter. Surface is a
    downward cone centered at the reference-orbit origin.
    """
    xs, ys, rtn = _sweep_grid(grid_extent_m, grid_steps, axis_pair)
    distances = jnp.linalg.norm(rtn, axis=-1)
    rewards = -distances
    vmin = float(rewards.min())
    return _render_sweep_surface(
        xs,
        ys,
        rewards,
        axis_pair,
        title=f"Lady-Bandit-Guard guard reward surface ({axis_pair.upper()} plane)",
        vmin=vmin,
        vmax=0.0,
        cmap="viridis",
        fig=fig,
    )


def plot_reward_position_sweep(
    cfg: Any,
    *,
    grid_extent_m: float = 2000.0,
    grid_steps: int = 64,
    axis_pair: str = "rt",
    t_seconds: float = 0.0,
    fig: Any | None = None,
) -> Any:
    """Dispatch a position-sweep surface based on ``cfg.game`` type."""
    from orbital_game.games.observation_blocking import ObservationBlocking
    from orbital_game.games.sun_blocking import SunBlocking

    game = getattr(cfg, "game", None)
    common = dict(grid_extent_m=grid_extent_m, grid_steps=grid_steps, axis_pair=axis_pair, fig=fig)
    if isinstance(game, PursuitEvasion):
        return plot_pursuit_evasion_position_sweep(cfg, **common)
    if isinstance(game, LadyBanditGuard):
        return plot_lady_bandit_guard_position_sweep(cfg, **common)
    if isinstance(game, SunBlocking):
        return plot_sun_blocking_position_sweep(cfg, t_seconds=t_seconds, **common)
    if isinstance(game, ObservationBlocking):
        return plot_observation_blocking_position_sweep(cfg, t_seconds=t_seconds, **common)
    raise TypeError(f"No position-sweep registered for game type {type(game).__name__}")


# ---------------------------------------------------------------------------
# Animated dual/triple-panel diagnostics — synchronize a 3D RTN scene with
# the cumulative reward (and, for OB, the elevation-vs-time curve). Useful
# for visually verifying that the reward fires only when geometry + gate
# conditions are correct.
# ---------------------------------------------------------------------------


def _reference_extent(*arrays: np.ndarray, padding: float = 1.2) -> float:
    """Return a symmetric scalar extent that contains all points in the arrays."""
    max_abs = 0.0
    for arr in arrays:
        if arr.size == 0:
            continue
        max_abs = max(max_abs, float(np.max(np.abs(arr))))
    if max_abs == 0.0:
        max_abs = 1.0
    return max_abs * padding


def _precompute_sb_animation(traj: Trajectory, cfg: Any):
    """Pre-compute per-step quantities needed by the SB animation."""
    import astrojax

    from orbital_game.games._frames import reference_orbit_eci_at_t, rtn_basis
    from orbital_game.games.sun_blocking import _epoch_from_mjd

    g_rtn = np.asarray(traj.env_state.guards.rtn[:, 0, :3])
    b_rtn = np.asarray(traj.env_state.bandits.rtn[:, 0, :3])
    times = np.asarray(traj.env_state.t)
    n_steps = g_rtn.shape[0]

    sun_dir_rtn = np.zeros((n_steps, 3))
    for i in range(n_steps):
        ref_pos, ref_vel = reference_orbit_eci_at_t(
            cfg.reference_orbit, cfg.epoch_mjd_utc, float(times[i])
        )
        rot = rtn_basis(ref_pos, ref_vel)  # R_rtn_to_eci
        epoch = _epoch_from_mjd(cfg.epoch_mjd_utc + float(times[i]) / 86400.0)
        sun_eci = astrojax.sun_position(epoch)
        diff = np.asarray(sun_eci) - np.asarray(ref_pos)
        diff = diff / (np.linalg.norm(diff) + 1e-12)
        sun_dir_rtn[i] = np.asarray(rot).T @ diff

    cum_g = np.cumsum(np.asarray(traj.sides.guard.reward))
    cum_b = np.cumsum(np.asarray(traj.sides.bandit.reward))

    return {
        "g_rtn": g_rtn,
        "b_rtn": b_rtn,
        "times": times,
        "sun_dir_rtn": sun_dir_rtn,
        "cum_g": cum_g,
        "cum_b": cum_b,
    }


def _precompute_ob_animation(traj: Trajectory, cfg: Any):
    """Pre-compute per-step quantities needed by the OB animation."""
    import astrojax

    from orbital_game.games._frames import reference_orbit_eci_at_t, rtn_basis, vehicle_eci_position
    from orbital_game.games.observation_blocking import (
        _epoch_from_mjd,
        target_visible_from_guard,
    )

    g_rtn = np.asarray(traj.env_state.guards.rtn[:, 0, :3])
    b_rtn = np.asarray(traj.env_state.bandits.rtn[:, 0, :3])
    times = np.asarray(traj.env_state.t)
    n_steps = g_rtn.shape[0]

    target_dir_rtn = np.zeros((n_steps, 3))
    visible = np.zeros(n_steps, dtype=bool)
    elevation_deg = np.zeros(n_steps)

    for i in range(n_steps):
        ref_pos, ref_vel = reference_orbit_eci_at_t(
            cfg.reference_orbit, cfg.epoch_mjd_utc, float(times[i])
        )
        rot = rtn_basis(ref_pos, ref_vel)
        epoch = _epoch_from_mjd(cfg.epoch_mjd_utc + float(times[i]) / 86400.0)
        eop = astrojax.zero_eop()
        rot_ecef_to_eci = astrojax.rotation_ecef_to_eci(eop, epoch)
        target_eci = np.asarray(rot_ecef_to_eci @ cfg.game.target_ecef_m)
        diff = target_eci - np.asarray(ref_pos)
        diff = diff / (np.linalg.norm(diff) + 1e-12)
        target_dir_rtn[i] = np.asarray(rot).T @ diff

        gv = jax.tree_util.tree_map(lambda x, idx=i: x[idx, 0], traj.env_state.guards)
        guard_eci = np.asarray(
            vehicle_eci_position(gv, cfg.reference_orbit, cfg.epoch_mjd_utc, times[i])
        )
        visible[i] = bool(
            target_visible_from_guard(
                jnp.asarray(target_eci), jnp.asarray(guard_eci), cfg.game.min_elevation_deg
            )
        )
        up = target_eci / (np.linalg.norm(target_eci) + 1e-12)
        dir_to_guard = (guard_eci - target_eci) / (np.linalg.norm(guard_eci - target_eci) + 1e-12)
        sin_el = float(np.dot(up, dir_to_guard))
        elevation_deg[i] = float(np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0))))

    cum_g = np.cumsum(np.asarray(traj.sides.guard.reward))
    cum_b = np.cumsum(np.asarray(traj.sides.bandit.reward))

    return {
        "g_rtn": g_rtn,
        "b_rtn": b_rtn,
        "times": times,
        "target_dir_rtn": target_dir_rtn,
        "visible": visible,
        "elevation_deg": elevation_deg,
        "cum_g": cum_g,
        "cum_b": cum_b,
    }


def plot_sun_blocking_animated_diagnostic(
    traj: Trajectory,
    cfg: Any,
    *,
    fps: int = 15,
    n_frames: int | None = None,
    fig: Any | None = None,
) -> tuple[Any, Any]:
    """Animated dual-panel SB diagnostic.

    Left panel: 3D RTN scene showing guard + bandit trajectories (trails to
    current frame), current positions as markers, and the sun-direction
    vector emanating from the guard with a star marker at d_target.

    Right panel: cumulative reward over time for both sides, with a moving
    vertical cursor at the current frame.

    Returns (Figure, matplotlib.animation.FuncAnimation). Save with
    ``anim.save("sb.mp4", fps=fps)`` or display via the IPython.display
    HTML5 video helper (``HTML(anim.to_jshtml())``) in a notebook.
    """
    import matplotlib.animation as manimation

    pre = _precompute_sb_animation(traj, cfg)
    g_rtn = pre["g_rtn"]
    b_rtn = pre["b_rtn"]
    times = pre["times"]
    sun_dir_rtn = pre["sun_dir_rtn"]
    cum_g = pre["cum_g"]
    cum_b = pre["cum_b"]
    n_steps = g_rtn.shape[0]
    d_target = float(cfg.game.target_viewing_distance_m)

    if n_frames is None:
        n_frames = n_steps
    indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)

    # Fixed scene extent including trajectories and the d_target star reach.
    extent = _reference_extent(g_rtn, b_rtn, np.array([[d_target, d_target, d_target]]))

    if fig is None:
        fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axr = fig.add_subplot(1, 2, 2)

    # Draw the cumulative reward once; animate only the cursor.
    axr.plot(times, cum_g, color="tab:blue", linewidth=2.0, label="guard cumulative reward")
    axr.plot(times, cum_b, color="tab:red", linewidth=2.0, label="bandit cumulative reward")
    axr.set_xlabel("time (s)")
    axr.set_ylabel("cumulative reward")
    axr.grid(True)
    axr.legend(loc="best", fontsize=8)
    axr.set_title("Sun-Blocking — cumulative reward")
    cursor = axr.axvline(times[0], color="black", linewidth=1.0, linestyle="--")

    def update(frame_idx: int):
        i = int(frame_idx)
        ax3d.cla()
        ax3d.plot(
            g_rtn[: i + 1, 0],
            g_rtn[: i + 1, 1],
            g_rtn[: i + 1, 2],
            color="tab:blue",
            linewidth=1.5,
            label="guard",
        )
        ax3d.plot(
            b_rtn[: i + 1, 0],
            b_rtn[: i + 1, 1],
            b_rtn[: i + 1, 2],
            color="tab:red",
            linewidth=1.5,
            label="bandit",
        )
        ax3d.scatter(
            [g_rtn[i, 0]], [g_rtn[i, 1]], [g_rtn[i, 2]], color="tab:blue", s=40, marker="o"
        )
        ax3d.scatter([b_rtn[i, 0]], [b_rtn[i, 1]], [b_rtn[i, 2]], color="tab:red", s=40, marker="o")

        # Sun-direction ray from the guard's current position.
        gp = g_rtn[i]
        sd = sun_dir_rtn[i]
        end = gp + 2.0 * d_target * sd
        ax3d.plot(
            [gp[0], end[0]],
            [gp[1], end[1]],
            [gp[2], end[2]],
            color="tab:orange",
            linewidth=1.5,
            label="sun direction",
        )
        star = gp + d_target * sd
        ax3d.scatter([star[0]], [star[1]], [star[2]], color="tab:orange", s=80, marker="*")

        ax3d.set_xlim(-extent, extent)
        ax3d.set_ylim(-extent, extent)
        ax3d.set_zlim(-extent, extent)
        ax3d.set_xlabel("R [m]")
        ax3d.set_ylabel("T [m]")
        ax3d.set_zlabel("N [m]")
        ax3d.set_title(f"SB scene — t = {times[i]:.0f} s")
        ax3d.view_init(elev=20, azim=45)
        ax3d.legend(loc="upper left", fontsize=7)

        cursor.set_xdata([times[i], times[i]])
        return (cursor,)

    anim = manimation.FuncAnimation(
        fig,
        update,
        frames=indices,
        interval=1000.0 / max(fps, 1),
        blit=False,
    )
    fig.tight_layout()
    return fig, anim


def plot_observation_blocking_animated_diagnostic(
    traj: Trajectory,
    cfg: Any,
    *,
    fps: int = 15,
    n_frames: int | None = None,
    fig: Any | None = None,
) -> tuple[Any, Any]:
    """Animated three-panel OB diagnostic.

    Left panel: 3D RTN scene showing trajectories + target-direction vector
    (with d_target star marker) + per-frame guard, bandit, and target
    indicator. The target-vector is dimmed when the gate is closed
    (target not visible from guard).

    Top-right panel: cumulative reward over time with moving cursor.

    Bottom-right panel: target elevation over time with the
    ``min_elevation_deg`` threshold line and shaded "invisible" windows.
    Cursor synced with top-right.

    Returns (Figure, FuncAnimation).
    """
    import matplotlib.animation as manimation

    pre = _precompute_ob_animation(traj, cfg)
    g_rtn = pre["g_rtn"]
    b_rtn = pre["b_rtn"]
    times = pre["times"]
    target_dir_rtn = pre["target_dir_rtn"]
    visible = pre["visible"]
    elevation_deg = pre["elevation_deg"]
    cum_g = pre["cum_g"]
    cum_b = pre["cum_b"]
    n_steps = g_rtn.shape[0]
    d_target = float(cfg.game.target_viewing_distance_m)
    min_el = float(cfg.game.min_elevation_deg)

    if n_frames is None:
        n_frames = n_steps
    indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)

    extent = _reference_extent(g_rtn, b_rtn, np.array([[d_target, d_target, d_target]]))

    if fig is None:
        fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_cum = fig.add_subplot(gs[0, 1])
    ax_el = fig.add_subplot(gs[1, 1], sharex=ax_cum)

    # Cumulative-reward panel (drawn once).
    ax_cum.plot(times, cum_g, color="tab:blue", linewidth=2.0, label="guard cum. reward")
    ax_cum.plot(times, cum_b, color="tab:red", linewidth=2.0, label="bandit cum. reward")
    ax_cum.set_ylabel("cumulative reward")
    ax_cum.grid(True)
    ax_cum.legend(loc="best", fontsize=8)
    ax_cum.set_title("Observation-Blocking — cumulative reward")
    cursor_cum = ax_cum.axvline(times[0], color="black", linewidth=1.0, linestyle="--")

    # Elevation panel (drawn once).
    ax_el.plot(times, elevation_deg, color="tab:green", linewidth=1.5, label="target elevation")
    ax_el.axhline(min_el, color="red", linestyle="--", label=f"min_elevation_deg ({min_el:g}°)")
    invisible_mask = (~visible).tolist()
    ax_el.fill_between(
        times,
        -90,
        90,
        where=invisible_mask,
        alpha=0.15,
        color="grey",
        step="mid",
        label="target not visible",
    )
    ax_el.set_xlabel("time (s)")
    ax_el.set_ylabel("elevation (°)")
    ax_el.grid(True)
    ax_el.legend(loc="best", fontsize=8)
    cursor_el = ax_el.axvline(times[0], color="black", linewidth=1.0, linestyle="--")

    def update(frame_idx: int):
        i = int(frame_idx)
        ax3d.cla()
        ax3d.plot(
            g_rtn[: i + 1, 0],
            g_rtn[: i + 1, 1],
            g_rtn[: i + 1, 2],
            color="tab:blue",
            linewidth=1.5,
            label="guard",
        )
        ax3d.plot(
            b_rtn[: i + 1, 0],
            b_rtn[: i + 1, 1],
            b_rtn[: i + 1, 2],
            color="tab:red",
            linewidth=1.5,
            label="bandit",
        )
        ax3d.scatter(
            [g_rtn[i, 0]], [g_rtn[i, 1]], [g_rtn[i, 2]], color="tab:blue", s=40, marker="o"
        )
        ax3d.scatter([b_rtn[i, 0]], [b_rtn[i, 1]], [b_rtn[i, 2]], color="tab:red", s=40, marker="o")

        # Target-direction ray from the guard's current position; dimmed when
        # the visibility gate is closed.
        gp = g_rtn[i]
        td = target_dir_rtn[i]
        end = gp + 2.0 * d_target * td
        if visible[i]:
            ray_color = "tab:orange"
            ray_style = "-"
            ray_label = "target direction (visible)"
        else:
            ray_color = "dimgray"
            ray_style = "--"
            ray_label = "target direction (not visible)"
        ax3d.plot(
            [gp[0], end[0]],
            [gp[1], end[1]],
            [gp[2], end[2]],
            color=ray_color,
            linestyle=ray_style,
            linewidth=1.5,
            label=ray_label,
        )
        star = gp + d_target * td
        ax3d.scatter([star[0]], [star[1]], [star[2]], color=ray_color, s=80, marker="*")

        ax3d.set_xlim(-extent, extent)
        ax3d.set_ylim(-extent, extent)
        ax3d.set_zlim(-extent, extent)
        ax3d.set_xlabel("R [m]")
        ax3d.set_ylabel("T [m]")
        ax3d.set_zlabel("N [m]")
        ax3d.set_title(f"OB scene — t = {times[i]:.0f} s")
        ax3d.view_init(elev=20, azim=45)
        ax3d.legend(loc="upper left", fontsize=7)

        cursor_cum.set_xdata([times[i], times[i]])
        cursor_el.set_xdata([times[i], times[i]])
        return (cursor_cum, cursor_el)

    anim = manimation.FuncAnimation(
        fig,
        update,
        frames=indices,
        interval=1000.0 / max(fps, 1),
        blit=False,
    )
    fig.tight_layout()
    return fig, anim
