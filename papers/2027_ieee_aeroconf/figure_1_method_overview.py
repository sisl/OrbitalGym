# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10", "numpy"]
# ///
"""Figure 1 of the paper: sampling an encounter, and the decision loop of an orbital game.

    uv run papers/2027_ieee_aeroconf/figure_1_method_overview.py

The left panel shows initial states. For lady-bandit-guard they are the 128
states that OrbitalGym's relative-ellipse sampler drew for two guards and one
bandit, stored in ``figure_data/method_overview.npz``; the inner guard ring is
drawn 2.5 times larger than sampled so that it separates from the lady. For
pursuit-evasion they are the first pursuer starts of the experiments, at the
300 m initial separation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#25313C"
GREY = "#6C7379"
PE_SEPARATION_M = 300.0
PE_SEED = 20260929
PE_SHOWN = 14


def pursuer_starts(count, seed, separation):
    """Pursuer start positions of the experiments: uniform directions at the separation."""
    direction = np.random.default_rng(seed).normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return separation * direction


def draw():
    data = np.load(ROOT / "figure_data" / "method_overview.npz")
    guard = data["guard_position_m"] / 1000
    bandit = data["bandit_position_m"] / 1000
    # The sampled supports are drift-free 2:1 ellipses of 0.3, 2, and 3 km radial amplitude.
    for positions, amplitudes in ((guard, np.array([0.3, 2.0])), (bandit, np.array([3.0]))):
        invariant = (positions[..., 0] / amplitudes) ** 2 + (
            positions[..., 1] / (2 * amplitudes)
        ) ** 2
        np.testing.assert_allclose(invariant, 1, atol=1e-5)
    shown_guard = guard.copy()
    shown_guard[:, 0, :] *= 0.75 / 0.3
    pursuer = pursuer_starts(128, PE_SEED, PE_SEPARATION_M)[:PE_SHOWN]

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
    fig = plt.figure(figsize=(7.16, 3.8), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")

    def text(x, y, s, size=8, weight="normal", color=INK, ha="center", **kw):
        return canvas.text(
            x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va="center", **kw
        )

    def box(x, y, w, h, label, color=INK, fill="white", size=8):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.007,rounding_size=0.012",
            facecolor=fill,
            edgecolor=color,
            linewidth=0.9,
        )
        canvas.add_patch(p)
        text(x + w / 2, y + h / 2, label, size, color=color, linespacing=1.5)
        return p

    def arrow(a, b, color=GREY, style="-", connection="arc3,rad=0", lw=0.9, heads="-|>"):
        canvas.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle=heads,
                mutation_scale=8,
                linewidth=lw,
                color=color,
                linestyle=style,
                connectionstyle=connection,
                shrinkA=0,
                shrinkB=0,
            )
        )

    canvas.add_patch(
        FancyBboxPatch(
            (0.014, 0.03),
            0.285,
            0.935,
            boxstyle="round,pad=0,rounding_size=.015",
            facecolor="#F7F9FB",
            edgecolor="#D5DCE2",
            lw=0.7,
        )
    )
    canvas.add_patch(
        FancyBboxPatch(
            (0.345, 0.03),
            0.642,
            0.935,
            boxstyle="round,pad=0,rounding_size=.015",
            facecolor="white",
            edgecolor="#D5DCE2",
            lw=0.7,
        )
    )
    text(0.157, 0.921, "1  Sample an encounter", 9.1, "bold")
    text(0.666, 0.921, "2  Decide, act, observe", 9.1, "bold")
    text(0.666, 0.862, "Both sides choose before the same joint step", 7.3)

    def style(ax, unit):
        ax.set_xlabel(rf"$x_T$ ({unit})", fontsize=7, labelpad=1)
        ax.set_ylabel(rf"$x_R$ ({unit})", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6.4, length=2, pad=1.5)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#ADB5BC")
            spine.set_linewidth(0.6)

    text(0.157, 0.866, "Lady-bandit-guard", 7.6, "bold")
    text(0.157, 0.834, "Lady (star) at origin; random orbit phases", 6.5)
    ax = fig.add_axes([0.07, 0.59, 0.215, 0.225], facecolor="#F7F9FB")
    phi = np.linspace(0, 2 * np.pi, 400)
    for a, ls in ((0.75, "-"), (2.0, "--")):
        ax.plot(2 * a * np.sin(phi), -a * np.cos(phi), color=BLUE, lw=0.8, ls=ls)
    ax.plot(6 * np.sin(phi), -3 * np.cos(phi), color=ORANGE, lw=0.8)
    for i in range(2):
        ax.scatter(
            shown_guard[:, i, 1], shown_guard[:, i, 0], s=2, color=BLUE, alpha=0.30, linewidths=0
        )
        ax.plot(
            *shown_guard[0, i, [1, 0]],
            marker="s" if i == 0 else "D",
            ms=4.2,
            color=BLUE,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax.scatter(bandit[:, 0, 1], bandit[:, 0, 0], s=2, color=ORANGE, alpha=0.30, linewidths=0)
    ax.plot(
        *bandit[0, 0, [1, 0]],
        marker="^",
        ms=5,
        color=ORANGE,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    ax.plot(0, 0, "*", ms=5.6, color=INK, zorder=5)
    ax.set(
        xlim=(-6.6, 6.6),
        ylim=(-3.4, 3.4),
        aspect="equal",
        xticks=[-6, 0, 6],
        yticks=[-3, 0, 3],
    )
    style(ax, "km")

    def marker(color, shape, size):
        return Line2D(
            [],
            [],
            color=color,
            marker=shape,
            ms=size,
            ls="",
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    def key(handles, labels, anchor):
        fig.legend(
            handles=handles,
            labels=labels,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
            loc="upper left",
            bbox_to_anchor=anchor,
            ncol=1,
            fontsize=6.2,
            frameon=False,
            handlelength=3.2,
            handletextpad=0.5,
            labelspacing=0.35,
            borderpad=0,
            labelcolor=INK,
        )

    key(
        [
            (
                marker(BLUE, "s", 3.6),
                marker(BLUE, "D", 3.4),
                marker(ORANGE, "^", 4.2),
                marker(INK, "*", 6.5),
            ),
            (Line2D([], [], color=BLUE, lw=0.8), Line2D([], [], color=ORANGE, lw=0.8)),
        ],
        ["Markers: true positions", "Rings: belief (unknown phase)"],
        (0.045, 0.522),
    )

    canvas.plot([0.035, 0.279], [0.447, 0.447], color="#D5DCE2", lw=0.7)
    text(0.157, 0.417, "Pursuit-evasion", 7.6, "bold")
    text(0.157, 0.385, "Uniform direction on a sphere; both at rest", 6.5)
    key(
        [marker(BLUE, "s", 3.6), Line2D([], [], color=ORANGE, marker="o", ms=3, alpha=0.6, ls="")],
        ["Evader: true position (origin)", "Pursuer: true starts (samples)"],
        (0.045, 0.362),
    )
    ax = fig.add_axes([0.07, 0.095, 0.215, 0.195], facecolor="#F7F9FB")
    # Pursuer starts are 3D points projected onto the radial/along-track plane.
    ax.scatter(pursuer[:, 1], pursuer[:, 0], s=9, color=ORANGE, alpha=0.6, linewidths=0)
    ax.plot(0, 0, "s", ms=4.2, color=BLUE, markeredgecolor="white", markeredgewidth=0.5)
    ax.set(
        xlim=(-360, 360),
        ylim=(-360, 360),
        aspect="equal",
        xticks=[-300, 0, 300],
        yticks=[-300, 0, 300],
    )
    style(ax, "m")
    arrow((0.304, 0.505), (0.342, 0.505), INK)
    text(0.323, 0.581, r"$s_0$", 8)

    for team, role, symbol, count, mid, color, fill in (
        ("Guard", "evader", "g", "N_g", 0.705, BLUE, "#F0F7FB"),
        ("Bandit", "pursuer", "b", "N_b", 0.445, ORANGE, "#FFF5EF"),
    ):
        text(0.596, mid + 0.115, f"{team} beliefs ({role} in PE)", 7.2, color=color)
        box(0.388, mid - 0.049, 0.105, 0.098, "Update each\nbelief", color, fill, 7.1)
        for idx, center in (("1", mid + 0.061), (count, mid - 0.061)):
            box(
                0.546,
                center - 0.03,
                0.086,
                0.06,
                rf"$b^{{{symbol}}}_{{{idx},k}}$",
                color,
                fill,
                8.6,
            )
            # Each observation updates its own belief; branching here does
            # not represent sharing observations between observers.
            canvas.plot([0.502, 0.516, 0.516], [mid, mid, center], color=color, lw=0.8)
            arrow((0.516, center), (0.537, center), color)
            canvas.plot([0.640, 0.755, 0.755], [center, center, mid], color=color, lw=0.8)
        text(0.567, mid, r"$\vdots$", 8, color=color)
        arrow((0.608, mid - 0.023), (0.608, mid + 0.023), color, style="--", lw=0.8, heads="<->")
        text(0.696, mid, "Exchange if\ncomms available", 6.4, color=color, linespacing=1.25)
        box(
            0.788,
            mid - 0.080,
            0.173,
            0.160,
            f"{team} policy " + rf"$\pi^{symbol}$" + "\ne.g. PD, MPPI,\nMCTS",
            color,
            fill,
            7.1,
        )
        arrow((0.755, mid), (0.779, mid), color)
    # Both action arrows meet at the same state transition; neither policy
    # observes the other team's newly selected action before choosing its own.
    canvas.plot([0.968, 0.977, 0.977], [0.705, 0.705, 0.195], color=BLUE, lw=0.9)
    arrow((0.977, 0.195), (0.949, 0.195), BLUE)
    arrow((0.8745, 0.358), (0.8745, 0.274), ORANGE)
    text(0.966, 0.320, r"$a^g_k$", 7.3, color=BLUE, ha="right")
    text(0.8515, 0.317, r"$a^b_k$", 7.3, color=ORANGE)
    box(
        0.741,
        0.125,
        0.200,
        0.140,
        "Propagate environment\n" + r"$s_k \rightarrow s_{k+1}$; $\Delta t=10$ s",
        INK,
        "#F5F6F7",
        7.6,
    )
    box(
        0.392,
        0.125,
        0.300,
        0.140,
        "Generate measurement observations\nfrom true state\n" + r"$s_{k+1} \rightarrow o_{k+1}$",
        INK,
        "white",
        7.0,
    )
    arrow((0.732, 0.195), (0.702, 0.195), INK)
    canvas.plot([0.383, 0.366, 0.366], [0.195, 0.195, 0.705], color=GREY, lw=0.9)
    arrow((0.366, 0.705), (0.379, 0.705), BLUE)
    arrow((0.366, 0.445), (0.379, 0.445), ORANGE)
    text(0.841, 0.091, r"Limit $\Delta v$; use fuel; check events", 6.6)
    text(0.666, 0.055, "Repeat until capture, breach, or timeout", 7.1)

    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"figure_1_method_overview.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
