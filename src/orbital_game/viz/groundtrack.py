"""Equirectangular ground-track plot with a cartopy basemap.

Renders:
  - Earth basemap (cartopy ``stock_img`` + coastlines)
  - Reference orbit ground track (ECI -> ECEF via brahe -> geodetic)
  - Optional per-side ground tracks (only meaningful when the per-side ECI
    diverges visibly from the reference; HCW-formation members within a
    few km of the reference look identical at this resolution and are
    therefore drawn as the reference track)
  - Ground-station markers + small-circle footprints at each station's
    elevation mask

`brahe` is a core dep; `cartopy` enters the env transitively via brahe.
EOP is initialized lazily on first call.
"""

from __future__ import annotations

from typing import Any

import brahe as _brahe
import cartopy.crs as ccrs
import jax
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_EOP_INITIALIZED = False


def _ensure_eop_initialized() -> None:
    """One-time EOP setup. brahe panics if accessed before initialization."""
    global _EOP_INITIALIZED
    if _EOP_INITIALIZED:
        return
    _brahe.set_global_eop_provider_from_static_provider(_brahe.StaticEOPProvider.from_zero())
    _EOP_INITIALIZED = True


def _eci_to_lonlat(eci_pos_m: np.ndarray, t_jd_utc: np.ndarray) -> np.ndarray:
    """Convert per-tick ECI positions (T, 3) to (T, 2) lon/lat in degrees.

    Loops over time outside JIT — viz is not on a hot path. Brahe's
    `position_ecef_to_geodetic` returns `[lon, lat, alt]` (note ordering)
    and accepts an explicit `AngleFormat`, so we ask for DEGREES directly.
    """
    _ensure_eop_initialized()
    out = np.empty((eci_pos_m.shape[0], 2))
    for i, (pos, jd) in enumerate(zip(eci_pos_m, t_jd_utc, strict=True)):
        # pyrefly: ignore[bad-argument-type]
        epc = _brahe.Epoch.from_jd(float(jd), _brahe.TimeSystem.UTC)
        ecef = _brahe.position_eci_to_ecef(epc, np.asarray(pos, dtype=np.float64))
        # pyrefly: ignore[bad-argument-type]
        geod = _brahe.position_ecef_to_geodetic(ecef, _brahe.AngleFormat.DEGREES)
        # geod = [lon_deg, lat_deg, alt_m]
        out[i] = (float(geod[0]), float(geod[1]))
    # Wrap longitudes to [-180, 180] for clean equirectangular plotting.
    out[:, 0] = ((out[:, 0] + 180.0) % 360.0) - 180.0
    return out


def _split_antimeridian(lonlat: np.ndarray, threshold_deg: float = 90.0) -> np.ndarray:
    """Insert NaN rows wherever consecutive longitudes jump beyond ``threshold_deg``.

    Cartopy ``PlateCarree`` will otherwise paint a horizontal line straight
    across the map at every antimeridian crossing. NaN-breaks let
    ``ax.plot`` draw separate segments per visible pass.

    The threshold defaults to ``90°``, well above the ~few-degrees-per-tick
    motion of a real LEO ground track at our typical ``dt`` (~10–30 s) but
    well below the ~360° apparent-jump produced when wrapping a longitude
    from +180° to -180°. Catching wraps near 180° but ALSO unphysical
    jumps in the 90°–180° band is needed because:

      * The wrap helper in ``_eci_to_lonlat`` collapses any out-of-range
        longitude into ``[-180, 180)``; depending on the exact float at the
        boundary the apparent jump can sit just below 180°.
      * Numerically-noisy or non-monotonic time inputs can produce
        physically-impossible large jumps that would otherwise paint a
        horizontal line across the map.

    A 180° threshold misses both classes; 90° is permissive enough for
    real motion (LEO ground speed ≈ 0.06°/s, so a 30-s tick advances <2°
    in longitude) and tight enough to catch the artifacts.
    """
    if lonlat.shape[0] < 2:
        return lonlat
    lon = lonlat[:, 0]
    dlon = np.abs(np.diff(lon))
    breaks = np.where(dlon > threshold_deg)[0]
    if breaks.size == 0:
        return lonlat
    # Build a copy with a NaN row inserted after each break index.
    out_rows = []
    last = 0
    nan_row = np.full((1, 2), np.nan)
    for b in breaks:
        out_rows.append(lonlat[last : b + 1])
        out_rows.append(nan_row)
        last = b + 1
    out_rows.append(lonlat[last:])
    return np.concatenate(out_rows, axis=0)


def _is_per_side_meaningful(side_eci: Any) -> bool:
    """``True`` when the user passed an actual per-side ECI history.

    HCW formation members live within km of the reference; on a global
    equirectangular map those offsets are below pixel resolution. The
    notebook (and tests) typically pass a broadcast of the reference orbit
    in their ``guard_eci`` / ``bandit_eci`` slot. When that's the case we
    skip the per-side overlay rather than draw a duplicate of the reference
    track. Callers that genuinely want per-side tracks can pass
    ``side_eci=None`` for "skip this side" or a divergent ECI history.
    """
    if side_eci is None:
        return False
    arr = np.asarray(side_eci)
    return not (arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 1)


def plot_groundtrack(
    *,
    epoch_mjd_utc: float,
    time_s: jax.Array,
    reference_orbit_eci: jax.Array,
    guard_eci: jax.Array | None = None,
    bandit_eci: jax.Array | None = None,
    guard_network: Any = None,
    bandit_network: Any = None,
    figsize: tuple[float, float] = (12.0, 6.0),
):
    """Build an equirectangular ground-track figure with an Earth basemap.

    Inputs are time-stacked ECI states (T, 6) for the reference orbit, and
    (T, N, 6) for each side. Per-side ECI histories that match the reference
    (e.g. broadcast HCW members) are not redrawn — they would land on top of
    the reference track at this scale. Networks may be ``None`` (that side's
    stations are omitted).

    Returns the matplotlib Figure for further customization or saving.
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()  # pyrefly: ignore[missing-attribute]
    # Best-effort basemap. ``stock_img`` requires the cartopy data file to be
    # bundled with the install; when missing we fall back to filled coastlines
    # so the plot still renders something map-like instead of an empty grid.
    try:
        ax.stock_img()  # pyrefly: ignore[missing-attribute]
    except Exception:  # noqa: BLE001 — cartopy raises a variety of fetch errors
        # pyrefly: ignore[missing-attribute]
        ax.add_feature(__import__("cartopy.feature", fromlist=["LAND"]).LAND, alpha=0.3)
    ax.coastlines(linewidth=0.5)  # pyrefly: ignore[missing-attribute]
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)  # pyrefly: ignore[missing-attribute]

    t_jd = epoch_mjd_utc + 2400000.5 + np.asarray(time_s) / 86400.0

    # Reference ground track — thick red line.
    ref_lonlat = _eci_to_lonlat(np.asarray(reference_orbit_eci[:, :3]), t_jd)
    ref_split = _split_antimeridian(ref_lonlat)
    ax.plot(
        ref_split[:, 0],
        ref_split[:, 1],
        color="red",
        linewidth=1.8,
        alpha=0.95,
        label="reference orbit",
        transform=ccrs.PlateCarree(),
    )

    # Optional per-side overlays. We draw them only when the caller supplies
    # arrays that visibly differ from the reference. See the module docstring.
    for side_eci, color, label in (
        (guard_eci, "tab:blue", "guards"),
        (bandit_eci, "tab:orange", "bandit"),
    ):
        if not _is_per_side_meaningful(side_eci):
            continue
        side_arr = np.asarray(side_eci)
        # Quick check: if the per-side ECI is bit-identical to a broadcast of
        # the reference, skip — it would just overdraw the red line.
        ref_arr = np.asarray(reference_orbit_eci)
        if side_arr.shape[0] == ref_arr.shape[0]:
            broadcast_ref = np.broadcast_to(ref_arr[:, None, :], side_arr.shape)
            if np.allclose(side_arr, broadcast_ref):
                continue
        n = side_arr.shape[1]
        for i in range(n):
            lonlat = _eci_to_lonlat(side_arr[:, i, :3], t_jd)
            lonlat_split = _split_antimeridian(lonlat)
            ax.plot(
                lonlat_split[:, 0],
                lonlat_split[:, 1],
                color=color,
                linewidth=0.9,
                alpha=0.7,
                label=f"{label}[{i}]" if i == 0 else None,
                transform=ccrs.PlateCarree(),
            )

    # Stations + footprints. Use ``ccrs.PlateCarree()`` as the data CRS so
    # cartopy reprojects markers & patches consistently.
    for net, color, side_label in (
        (guard_network, "tab:blue", "guard"),
        (bandit_network, "tab:orange", "bandit"),
    ):
        if net is None:
            continue
        for j, st in enumerate(net.stations):
            lat = float(st.lat_deg)
            lon = float(st.lon_deg)
            ax.plot(
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
            # Footprint approximation: small-circle radius at the elevation mask.
            # cos(gamma) = R_earth / (R_earth + h) * cos(elevation_mask)
            # where gamma is the half-angle subtended at Earth's center between
            # the station and the visible horizon at orbit altitude h. Stand-in
            # h = 500 km; the result renders as an ellipse in lon/lat (correct
            # only at the equator). Illustrative, not navigation-grade.
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
            ax.add_patch(ellipse)

    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    return fig
