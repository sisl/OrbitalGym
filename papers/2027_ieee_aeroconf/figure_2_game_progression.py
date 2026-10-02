# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10", "numpy"]
# ///
"""Figure 2 of the paper: four snapshots of one lady-bandit-guard encounter.

    uv run papers/2027_ieee_aeroconf/figure_2_game_progression.py

The encounter was simulated with OrbitalGym's reference lady-bandit-guard
configuration: particle-filter beliefs, 30 degree cone sensors, a 50 m capture
radius, and a 20 m breach radius. ``figure_data/game_progression.npz`` holds
the true trajectories and, at the four snapshot times, each side's particles
about the other and its attitude. Panels show the radial and along-track
components; the cross-track component of this encounter is zero.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Wedge

ROOT = Path(__file__).resolve().parent
BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#25313C"
RED = "#C43C39"
GREY = "#6C7379"


def callout(ax, text, xy, xytext, ha="left", arrow=True):
    """Small annotation naming one modeling choice of the scenario."""
    ax.annotate(
        text,
        xy,
        xytext=xytext,
        textcoords="data",
        ha=ha,
        va="center",
        fontsize=5.9,
        color=INK,
        linespacing=1.2,
        arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.6, "shrinkA": 1, "shrinkB": 1.5}
        if arrow
        else None,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#C9D0D6",
            "lw": 0.5,
            "alpha": 0.92,
        },
        zorder=12,
    )


def draw():
    a = np.load(ROOT / "figure_data" / "game_progression.npz")
    indices = a["snapshot_indices"].tolist()
    half_angle_deg = float(a["sensor_half_angle_deg"])
    guard_ring_m = float(a["guard_ring_m"])
    prior_sigma_m = float(a["guard_prior_sigma_m"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.255, top=0.965, wspace=0.25, hspace=0.52)
    for j, (ax, k) in enumerate(zip(axes.flat, indices, strict=True)):
        zoom = j == 3
        unit = 1 if zoom else 1000
        seen = {}
        for side, color, marker, ls in [("guard", BLUE, "s", "-"), ("bandit", ORANGE, "^", "--")]:
            true = a[f"{side}_position_m"][: k + 1]
            ax.plot(true[:, 1] / unit, true[:, 0] / unit, color=color, ls=ls, lw=1.15, zorder=2)
            # Each side tracks the other side in slot1. Colour by the estimated
            # vehicle, so orange always denotes bandit and blue always guard.
            target = "bandit" if side == "guard" else "guard"
            target_color = ORANGE if target == "bandit" else BLUE
            points = a[f"{side}_particles_m"][j]
            w = a[f"{side}_weights"][j]
            ax.scatter(
                points[:, 1] / unit,
                points[:, 0] / unit,
                s=700 * w,
                color=target_color,
                alpha=0.42,
                edgecolors="none",
                zorder=3,
            )
            ax.plot(
                true[-1, 1] / unit,
                true[-1, 0] / unit,
                marker=marker,
                ms=5.2,
                color=color,
                mec="white",
                mew=0.7,
                zorder=9,
            )
            # Exact intersection of the 3D sensor cone with the RT plane.
            # Its finite drawing length is clipped to the axes, not sensor range.
            quat = a[f"{side}_quaternion_wxyz"][j].astype(float)
            quat /= np.linalg.norm(quat)
            qw, qx, qy, qz = quat
            boresight = np.array(
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw)]
            )
            boresight /= np.linalg.norm(boresight)
            planar_length = np.linalg.norm(boresight[:2])
            half_angle = np.deg2rad(half_angle_deg)
            if planar_length >= np.cos(half_angle):
                half_slice = np.rad2deg(np.arccos(np.cos(half_angle) / planar_length))
                heading = np.rad2deg(np.arctan2(boresight[0], boresight[1]))
                ax.add_patch(
                    Wedge(
                        (true[-1, 1] / unit, true[-1, 0] / unit),
                        15000 / unit,
                        heading - half_slice,
                        heading + half_slice,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.09,
                        linewidth=0.6,
                        zorder=0,
                    )
                )
            displacement = a[f"{target}_position_m"][k] - true[-1]
            visible = bool(
                np.dot(boresight, displacement / np.linalg.norm(displacement)) >= np.cos(half_angle)
            )
            seen[side] = "yes" if visible else "no"
        ax.plot(0, 0, "*", ms=6.5, color=INK, zorder=7)
        ax.set_aspect("equal", adjustable="box")
        if zoom:
            p = a["guard_position_m"][k]
            ax.add_patch(Circle((p[1], p[0]), 50, fill=False, lw=0.8, ls=":", ec=BLUE, zorder=1))
            ax.add_patch(Circle((0, 0), 20, fill=False, lw=0.85, ls="--", ec=RED, zorder=4))
            ax.annotate(
                "Bandit breach\n20 m",
                (-14.14, 14.14),
                xytext=(-90, 40),
                textcoords="data",
                arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.6},
                fontsize=7,
                color=RED,
            )
            ax.set(xlim=(-100, 200), ylim=(-90, 90))
            separation = float(np.linalg.norm(a["bandit_position_m"][k] - p))
            ax.text(
                0.025,
                0.97,
                f"Separation {separation:.1f} m",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            )
            ax.annotate(
                "Guard capture\n50 m",
                (p[1] + 50, p[0]),
                xytext=(105, 40),
                textcoords="data",
                arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.6},
                fontsize=7,
                color=BLUE,
            )
        else:
            ax.set(xlim=(-1050 / 1000, 4850 / 1000), ylim=(-2550 / 1000, 650 / 1000))
        if j == 0:
            b0 = a["bandit_position_m"][k]
            separation = float(np.linalg.norm(b0))
            ax.plot(
                [0, b0[1] / unit], [0, b0[0] / unit], color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=1
            )
            heading = np.rad2deg(np.arctan2(b0[0], b0[1]))
            ax.text(
                0.5 * b0[1] / unit + 0.08,
                0.5 * b0[0] / unit + 0.17,
                f"Initial separation {separation / 1000:.1f} km",
                rotation=heading,
                rotation_mode="anchor",
                ha="center",
                va="bottom",
                fontsize=5.9,
                color=INK,
                zorder=12,
            )
            callout(
                ax,
                f"Guard relative orbit, {guard_ring_m:.0f} m;\nbandit's prior: phase unknown",
                (-0.28, -0.27),
                (-0.95, -0.95),
            )
            callout(
                ax,
                f"Guard's prior on bandit:\nGaussian, $\\sigma$ = {prior_sigma_m:.0f} m",
                (4.0, -2.12),
                (3.72, -2.2),
                ha="right",
            )
            callout(
                ax,
                f"Cone sensors (shaded):\n{half_angle_deg:.0f}\u00b0 half-angle, no range limit",
                (3.45, -0.35),
                (0.75, 0.33),
            )
            callout(
                ax,
                "HCW dynamics about a\ncircular 7000 km orbit",
                (0, 0),
                (-0.95, -2.2),
                arrow=False,
            )
        ax.text(
            -0.13,
            1.03,
            f"({chr(97 + j)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=INK,
        )
        units = "m" if zoom else "km"
        ax.set_xlabel(r"Along-track, $x_T$ (" + units + ")", fontsize=8, labelpad=2)
        ax.set_ylabel(r"Radial, $x_R$ (" + units + ")", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=7.2, length=2.2, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#ADB5BC")
        ax.text(
            0.5,
            -0.255,
            f"t = {int(a['time_s'][k])} s. Guard sees bandit: {seen['guard']}; "
            f"bandit sees guard: {seen['bandit']}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.9,
            color=INK,
        )
    handles = [
        Line2D(
            [], [], color=BLUE, marker="s", ms=4, lw=1.1, label="Guard: true position and history"
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            marker="^",
            ms=4,
            lw=1.1,
            ls="--",
            label="Bandit: true position and history",
        ),
        Line2D(
            [],
            [],
            color=BLUE,
            marker="o",
            ls="",
            ms=4,
            alpha=0.5,
            label="Bandit's particles about the guard",
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            marker="o",
            ls="",
            ms=4,
            alpha=0.5,
            label="Guard's particles about the bandit",
        ),
        Patch(facecolor=INK, alpha=0.12, label="Shading: sensor field of view"),
        Line2D([], [], color=INK, marker="*", ls="", ms=5, label="Lady (reference-frame origin)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.025),
        ncol=2,
        fontsize=7.2,
        frameon=False,
        columnspacing=1.7,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.7,
    )
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"figure_2_game_progression.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
