"""Tests for reward position-sweep surface plots."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from orbitalgym.games.lady_bandit_guard import make_lady_bandit_guard  # noqa: E402
from orbitalgym.games.observation_blocking import make_observation_blocking  # noqa: E402
from orbitalgym.games.pursuit_evasion import make_pursuit_evasion  # noqa: E402
from orbitalgym.games.sun_blocking import make_sun_blocking  # noqa: E402
from orbitalgym.viz.reward_diagnostics import (  # noqa: E402
    plot_lady_bandit_guard_position_sweep,
    plot_observation_blocking_position_sweep,
    plot_pursuit_evasion_position_sweep,
    plot_reward_position_sweep,
    plot_sun_blocking_position_sweep,
)


def test_sb_sweep_runs_and_has_axes():
    cfg = make_sun_blocking()
    fig = plot_sun_blocking_position_sweep(cfg, grid_steps=16)
    assert len(fig.axes) >= 1  # at least the 3D axes (colorbar adds another)
    plt.close(fig)


def test_ob_sweep_runs_and_has_axes():
    cfg = make_observation_blocking()
    fig = plot_observation_blocking_position_sweep(cfg, grid_steps=16)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_pe_sweep_runs():
    cfg = make_pursuit_evasion()
    fig = plot_pursuit_evasion_position_sweep(cfg, grid_steps=16)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_lbg_sweep_runs():
    cfg = make_lady_bandit_guard()
    fig = plot_lady_bandit_guard_position_sweep(cfg, grid_steps=16)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_dispatcher_dispatches():
    """plot_reward_position_sweep dispatches based on cfg.game type."""
    for cfg in (
        make_lady_bandit_guard(),
        make_pursuit_evasion(),
        make_sun_blocking(),
        make_observation_blocking(),
    ):
        fig = plot_reward_position_sweep(cfg, grid_steps=8)
        assert len(fig.axes) >= 1
        plt.close(fig)


def test_sb_sweep_peak_at_d_target_along_sun_axis():
    """The maximum reward in the sweep should land near +d_target from the guard,
    along the projected sun direction in RTN. We don't compute the projected
    sun direction here; instead, we just confirm the surface has both a peak
    near +1 and a trough near -1 at some grid cell."""
    cfg = make_sun_blocking(target_viewing_distance_m=500.0, range_decay_coef=4e-6)
    fig = plot_sun_blocking_position_sweep(cfg, grid_extent_m=1000.0, grid_steps=33)
    plt.close(fig)
    # Recompute Z values directly via the kernel for a deterministic check.
    import astrojax
    import jax

    from orbitalgym.games._frames import reference_orbit_eci_at_t, rtn_basis
    from orbitalgym.games.sun_blocking import _epoch_from_mjd, sun_blocking_kernel

    ref_pos, ref_vel = reference_orbit_eci_at_t(cfg.reference_orbit, cfg.epoch_mjd_utc, 0.0)
    rot = rtn_basis(ref_pos, ref_vel)
    guard_eci = ref_pos
    sun_eci = astrojax.sun_position(_epoch_from_mjd(cfg.epoch_mjd_utc))
    grid = jnp.linspace(-1000.0, 1000.0, 33)
    xs, ys = jnp.meshgrid(grid, grid, indexing="ij")
    rtn = jnp.zeros((33, 33, 3)).at[..., 0].set(xs).at[..., 1].set(ys)
    rtn_flat = rtn.reshape(-1, 3)
    bandit_eci_flat = ref_pos[None, :] + rtn_flat @ rot.T
    rewards = jax.vmap(
        lambda b: sun_blocking_kernel(
            guard_eci=guard_eci,
            bandit_eci=b,
            sun_eci=sun_eci,
            target_viewing_distance_m=cfg.game.target_viewing_distance_m,
            range_decay_coef=cfg.game.range_decay_coef,
        )
    )(bandit_eci_flat).reshape(33, 33)
    assert float(rewards.max()) > 0.5
    assert float(rewards.min()) < -0.5
