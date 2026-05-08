"""Combined 3D RTN scene + ground-track map figure.

Builds a 2-row figure that shares the same `frame` index between the
existing `RolloutScene` 3D render and the ground-track map. Used by
`interactive_viewer` and `save_animation` to step both panels in lockstep.

v1 implementation: the ground-track is rendered statically (full-episode
view); the 3D scene advances per frame. A future follow-up could add a
moving sub-satellite-point marker on the ground-track for true panel sync.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from orbital_game.viz.animation import RolloutScene, render_frame
from orbital_game.viz.groundtrack import (
    _eci_to_lonlat,
    _is_per_side_meaningful,
    _split_antimeridian,
)


def make_compound_figure(
    scene: RolloutScene,
    *,
    epoch_mjd_utc: float,
    reference_orbit_eci: Any,
    guard_eci: Any,
    bandit_eci: Any,
    guard_network: Any = None,
    bandit_network: Any = None,
    figsize: tuple[float, float] = (14.0, 10.0),
):
    """Builds a 2-row figure: 3D scene on top, ground-track on bottom.

    The 3D axes is named `ax_scene`; the map axes is named `ax_map` (a cartopy
    ``GeoAxes`` so coastlines, the Earth basemap, and lon/lat labels render).
    Caller advances the frame by re-rendering `render_frame(scene, ax_scene,
    frame)` while leaving `ax_map` static.

    Returns `(fig, ax_scene, ax_map)`.
    """
    fig = plt.figure(figsize=figsize)
    if scene.resolved_mode == "3d":
        ax_scene = fig.add_subplot(2, 1, 1, projection="3d")
    else:
        ax_scene = fig.add_subplot(2, 1, 1)
    ax_map = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())

    render_frame(scene, ax_scene, 0)

    ax_map.set_global()  # pyrefly: ignore[missing-attribute]
    try:
        ax_map.stock_img()  # pyrefly: ignore[missing-attribute]
    except Exception:  # noqa: BLE001
        import cartopy.feature as cfeature

        ax_map.add_feature(cfeature.LAND, alpha=0.3)  # pyrefly: ignore[missing-attribute]
    ax_map.coastlines(linewidth=0.5)  # pyrefly: ignore[missing-attribute]
    ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)  # pyrefly: ignore[missing-attribute]

    time_s = scene.traj.env_state.t
    t_jd = epoch_mjd_utc + 2400000.5 + np.asarray(time_s) / 86400.0
    ref_arr = np.asarray(reference_orbit_eci)
    ref_lonlat = _eci_to_lonlat(ref_arr[:, :3], t_jd)
    ref_split = _split_antimeridian(ref_lonlat)
    ax_map.plot(
        ref_split[:, 0],
        ref_split[:, 1],
        color="red",
        linewidth=1.8,
        alpha=0.95,
        label="reference orbit",
        transform=ccrs.PlateCarree(),
    )

    for side_eci, color, label in (
        (guard_eci, "tab:blue", "guards"),
        (bandit_eci, "tab:orange", "bandit"),
    ):
        if not _is_per_side_meaningful(side_eci):
            continue
        side_arr = np.asarray(side_eci)
        if side_arr.shape[0] == ref_arr.shape[0]:
            broadcast_ref = np.broadcast_to(ref_arr[:, None, :], side_arr.shape)
            if np.allclose(side_arr, broadcast_ref):
                continue
        n = side_arr.shape[1]
        for i in range(n):
            lonlat = _eci_to_lonlat(side_arr[:, i, :3], t_jd)
            lonlat_split = _split_antimeridian(lonlat)
            ax_map.plot(
                lonlat_split[:, 0],
                lonlat_split[:, 1],
                color=color,
                linewidth=0.9,
                alpha=0.7,
                label=f"{label}[{i}]" if i == 0 else None,
                transform=ccrs.PlateCarree(),
            )

    for net, color, side_label in (
        (guard_network, "tab:blue", "guard"),
        (bandit_network, "tab:orange", "bandit"),
    ):
        if net is None:
            continue
        for j, st in enumerate(net.stations):
            lat = float(st.lat_deg)
            lon = float(st.lon_deg)
            ax_map.plot(
                lon,
                lat,
                marker="^",
                color=color,
                markersize=10,
                markeredgecolor="black",
                transform=ccrs.PlateCarree(),
                linestyle="None",
                label=f"{side_label} stations" if j == 0 else None,
            )
            mask_rad = float(np.radians(float(st.elevation_mask_deg)))
            R_e = 6378137.0  # noqa: N806 - Earth radius (physics convention)
            h = 500e3
            cos_gamma = R_e / (R_e + h) * np.cos(mask_rad)
            gamma_rad = float(np.arccos(np.clip(cos_gamma, -1.0, 1.0)))
            radius_deg = float(np.degrees(gamma_rad))
            ellipse = mpatches.Ellipse(
                (lon, lat),
                width=2 * radius_deg / max(np.cos(np.radians(lat)), 0.1),
                height=2 * radius_deg,
                facecolor=color,
                alpha=0.18,
                edgecolor=color,
                linewidth=0.8,
                transform=ccrs.PlateCarree(),
            )
            ax_map.add_patch(ellipse)

    ax_map.legend(loc="lower left", fontsize=8, framealpha=0.85)
    return fig, ax_scene, ax_map
