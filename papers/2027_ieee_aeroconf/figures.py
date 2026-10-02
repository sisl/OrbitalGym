# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10", "numpy", "pandas>=2.2", "scienceplots>=2.2"]
# ///
"""Figures 3 to 7 and Table 6 of the paper, from the summaries of the experiment runs.

    uv run papers/2027_ieee_aeroconf/figures.py
    uv run papers/2027_ieee_aeroconf/figures.py figure_3_capability table_6_search

Each figure reads ``summary/<run>.csv``, which the experiment scripts write
next to this script, and writes a PDF and a PNG to ``figures/``. Win fractions
are plotted in percent. Outcomes that favor the bandit (the pursuer in
pursuit-evasion, the attacker in lady-bandit-guard) shade red and outcomes
that favor the guard shade blue.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401 - registers the "science" styles
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "summary"
OUT = ROOT / "figures"
FULL_WIDTH = 7.16

FEEDBACK = ("direct", "pd", "lqr", "hcw")
MPPI_PURSUERS = ("mppi_coast", "mppi_flee")
MPPI_GUARDS = ("mppi_coast", "mppi_approach")
LABELS = {
    "coast": "Coast",
    "direct": "Direct",
    "pd": "PD",
    "lqr": "LQR",
    "hcw": "HCW intercept",
    "mppi_coast": "MPPI-Coast",
    "mppi_flee": "MPPI-Flee",
    "random": "Random",
    "flee": "Flee",
    "transverse": "Transverse",
    "mppi_direct": "MPPI-Direct",
    "mppi_lqr": "MPPI-LQR",
    "mcts": "MCTS",
}
# Okabe-Ito colors, fixed per policy across the figures.
STYLE = {
    "lqr": ("#0072B2", "o"),
    "hcw": ("#009E73", "^"),
    "mppi_coast": ("#CC79A7", "D"),
    "mppi_flee": ("#D55E00", "P"),
}
GROUP_TITLES = ("Best feedback-control policy", "Best MPPI policy", "MPPI minus feedback control")
BANDIT_WINS = "RdBu_r"
GUARD_WINS = "RdBu"
TOWARD_BANDIT = "PuOr_r"
LEVELS = np.linspace(0.0, 100.0, 11)
GAIN_LEVELS = np.array([-100.0, -70.0, -50.0, -30.0, -10.0, 10.0, 30.0, 50.0, 70.0, 100.0])


def load(run):
    """One row per configuration; ``outcome`` is the win fraction of the pursuer or the guard."""
    return pd.read_csv(SUMMARY / f"{run}.csv")


def pe_ratios(frame):
    """Evader-over-pursuer thrust and budget ratios, the guard-over-bandit ratios of the paper."""
    frame = frame.copy()
    frame["thrust_ratio"] = (frame.evader_cap_mps / frame.pursuer_cap_mps).round(3)
    frame["budget_ratio"] = (frame.evader_budget_mps / frame.pursuer_budget_mps).round(3)
    return frame


def lbg_ratios(frame):
    """Guard-over-bandit thrust and budget ratios."""
    frame = frame.copy()
    frame["thrust_ratio"] = (frame.guard_cap_mps / frame.bandit_cap_mps).round(3)
    frame["budget_ratio"] = (frame.guard_budget_mps / frame.bandit_budget_mps).round(3)
    return frame


def best_of(frame, policies, keys, column="pursuer"):
    """Win fraction of the best policy of a group in each configuration, as a table."""
    return frame[frame[column].isin(policies)].groupby(keys).outcome.max().unstack()


def style():
    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def letter(ax, index):
    ax.text(
        -0.02,
        1.02,
        f"({'abcdefghij'[index]})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )


def titled(ax, index, title=None):
    """Panel letter, followed by the column title on the top row."""
    label = f"({'abcdefghij'[index]})"
    ax.set_title(f"{label} {title}" if title else label, fontsize=8, loc="left")


def contour_panel(ax, x, y, table, gain=False, cmap=BANDIT_WINS):
    """Filled contours, in percent, of a (y by x) table of win fractions.

    Outcome maps mark the labeled 50% contour.
    """
    xs, ys = np.meshgrid(x, y)
    values = 100.0 * np.clip(table, -1.0, 1.0)
    if gain:
        norm = TwoSlopeNorm(0.0, -100.0, 100.0)
        return ax.contourf(xs, ys, values, levels=GAIN_LEVELS, cmap=cmap, norm=norm)
    image = ax.contourf(xs, ys, values, levels=LEVELS, cmap=cmap)
    if np.nanmin(values) < 50.0 < np.nanmax(values):
        line = ax.contour(xs, ys, values, levels=[50.0], colors="black", linewidths=0.6)
        ax.clabel(line, fmt={50.0: "50%"}, fontsize=6, inline=True)
    return image


def ratio_axes(ax, xlabel, ylabel, ticks):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in ticks]))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def group_panels(frame, keys):
    """Best feedback-control policy, best MPPI policy, and their difference, as tables."""
    feedback = best_of(frame, FEEDBACK, keys)
    mppi = best_of(frame, MPPI_PURSUERS, keys)
    return (feedback, False), (mppi, False), (mppi - feedback, True)


def figure_3_capability():
    """Capture of the fleeing evader over the thrust and budget ratios, by pursuer budget."""
    d = pe_ratios(pd.concat([load("pe_1_fine"), load("pe_2_fine")]))
    budgets = sorted(d.pursuer_budget_mps.unique())
    fig, axes = plt.subplots(
        len(budgets), 3, figsize=(FULL_WIDTH, 6.0), sharex=True, sharey=True, layout="constrained"
    )
    for row, budget in enumerate(budgets):
        sub = d[d.pursuer_budget_mps == budget]
        for col, (table, gain) in enumerate(group_panels(sub, ["thrust_ratio", "budget_ratio"])):
            ax = axes[row, col]
            image = contour_panel(
                ax,
                table.columns.values,
                table.index.values,
                table.values,
                gain,
                TOWARD_BANDIT if gain else BANDIT_WINS,
            )
            ratio_axes(
                ax,
                r"Budget ratio $\rho_{\Delta v}$" if row == len(budgets) - 1 else None,
                f"$\\Delta v_B={budget:g}$ m/s\n" + r"Thrust ratio $\rho_u$" if col == 0 else None,
                (0.25, 0.5, 1.0, 2.0),
            )
            titled(ax, 3 * row + col, GROUP_TITLES[col] if row == 0 else None)
            if row == 0:
                label = "Change in capture fraction (%)" if gain else "Capture fraction (%)"
                fig.colorbar(image, ax=axes[:, col], shrink=0.6, label=label)
    save(fig, "figure_3_capability")


def figure_4_lbg_capability():
    """Defense by an idle guard and by the best guard of each group, against two bandits."""
    d = lbg_ratios(load("lbg_1_fine"))
    titles = ("Coasting guard", *GROUP_TITLES[:2])
    bandits = ("lqr", "mppi_lqr")
    fig, axes = plt.subplots(
        len(bandits), 3, figsize=(FULL_WIDTH, 4.3), sharex=True, sharey=True, layout="constrained"
    )
    for row, bandit in enumerate(bandits):
        sub = d[d.bandit == bandit]
        for col, guards in enumerate((("coast",), FEEDBACK, MPPI_GUARDS)):
            table = best_of(sub, guards, ["thrust_ratio", "budget_ratio"], "guard")
            ax = axes[row, col]
            image = contour_panel(
                ax, table.columns.values, table.index.values, table.values, cmap=GUARD_WINS
            )
            ratio_axes(
                ax,
                r"Budget ratio $\rho_{\Delta v}$" if row == len(bandits) - 1 else None,
                f"{LABELS[bandit]} bandit\n" + r"Thrust ratio $\rho_u$" if col == 0 else None,
                (0.5, 1.0, 2.0, 4.0),
            )
            titled(ax, 3 * row + col, titles[col] if row == 0 else None)
    fig.colorbar(image, ax=axes, shrink=0.6, label="Defense fraction (%)")
    save(fig, "figure_4_lbg_capability")


def figure_5_geometry():
    """Capture over initial separation and horizon against a coasting and a fleeing evader."""
    d = load("pe_3_fine")
    rows = ("coast", "flee")
    separations = (100, 200, 400, 800, 1600, 3200)
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(FULL_WIDTH, 4.3), sharex=True, sharey=True, layout="constrained"
    )
    for row, evader in enumerate(rows):
        sub = d[d.evader == evader]
        for col, (table, gain) in enumerate(group_panels(sub, ["separation_m", "horizon_s"])):
            ax = axes[row, col]
            image = contour_panel(
                ax,
                table.columns.values,
                table.index.values,
                table.values,
                gain,
                TOWARD_BANDIT if gain else BANDIT_WINS,
            )
            ax.set_yscale("log", base=2)
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(separations))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FixedFormatter([str(v) for v in separations])
            )
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xticks([300, 900, 1500])
            ax.minorticks_off()
            if row == len(rows) - 1:
                ax.set_xlabel("Game horizon $T$ (s)")
            if col == 0:
                ax.set_ylabel(f"{LABELS[evader]} evader\nSeparation $d_0$ (m)")
            titled(ax, 3 * row + col, GROUP_TITLES[col] if row == 0 else None)
            if row == 0 and col in (0, 2):
                label = "Change in capture fraction (%)" if gain else "Capture fraction (%)"
                fig.colorbar(
                    image, ax=axes[:, col] if col == 2 else axes[:, :2], shrink=0.6, label=label
                )
    save(fig, "figure_5_geometry")


# (sensor, range in m, noise fraction, tick label)
SENSOR_ORDER = (
    ("full", np.inf, 0.01, "Full"),
    ("radial", 5000.0, 0.01, "Radial\n5 km"),
    ("radial", 500.0, 0.01, "Radial\n500 m"),
    ("cone", 5000.0, 0.01, "Cone\n5 km"),
    ("cone", 500.0, 0.01, "Cone\n500 m"),
    ("radial", 5000.0, 0.1, "Noisy\nradial"),
)
# (evader, separation in m, thrust ratio, budget ratio)
INFORMATION_PANELS = (
    ("coast", 1000.0, 1.0, 0.75),
    ("flee", 300.0, 0.5, 1.0),
    ("flee", 300.0, 1.0, 0.75),
)


def figure_6_information():
    """Capture fraction by sensor for four pursuers in three settings."""
    d = pe_ratios(load("pe_4"))
    pursuers = ("lqr", "hcw", "mppi_coast", "mppi_flee")
    fig, axes = plt.subplots(
        1, len(INFORMATION_PANELS), figsize=(FULL_WIDTH, 2.5), sharey=True, layout="constrained"
    )
    for index, (ax, (evader, separation, thrust, fuel)) in enumerate(
        zip(axes, INFORMATION_PANELS, strict=True)
    ):
        sub = d[
            (d.evader == evader)
            & (d.separation_m == separation)
            & (d.thrust_ratio == thrust)
            & (d.budget_ratio == fuel)
        ]
        offsets = np.linspace(-0.24, 0.24, len(pursuers))
        for offset, pursuer in zip(offsets, pursuers, strict=True):
            rates = []
            for sensor, max_range, noise, _ in SENSOR_ORDER:
                cell = sub[(sub.pursuer == pursuer) & (sub.sensor == sensor)]
                if sensor != "full":
                    cell = cell[
                        (cell.sensor_range_m == max_range) & np.isclose(cell.noise_fraction, noise)
                    ]
                rates.append(100.0 * cell.outcome.mean())
            color, marker = STYLE[pursuer]
            ax.plot(
                np.arange(len(SENSOR_ORDER)) + offset,
                rates,
                marker=marker,
                color=color,
                linestyle="none",
                markersize=4,
                label=LABELS[pursuer],
            )
        ax.set_xticks(range(len(SENSOR_ORDER)), [s[3] for s in SENSOR_ORDER], fontsize=6.5)
        ax.set_ylim(-3.0, 103.0)
        ax.minorticks_off()
        ax.set_xlabel(
            f"{LABELS[evader]} evader, $d_0={separation:g}$ m, "
            f"$\\rho_u={thrust:g}$, $\\rho_{{\\Delta v}}={fuel:g}$",
            fontsize=7.5,
        )
        letter(ax, index)
    axes[0].set_ylabel("Capture fraction (%)")
    axes[-1].legend(loc="upper right", frameon=True, fontsize=7)
    save(fig, "figure_6_information")


def figure_7_lookahead():
    """MPPI-Flee capture fraction by planning lookahead for three levels of pursuer information."""
    d = load("pe_5")
    d["lookahead_s"] = d.planner_segments * d.planner_repeat * d.dt_s
    levels = (
        ("full", None, "Full information", "-", "o", "#0072B2"),
        ("radial", 1e-3, "Radial, 0.1% noise", "--", "s", "#E69F00"),
        ("radial", 1e-2, "Radial, 1% noise", ":", "^", "#D55E00"),
    )
    evaders = ("flee", "mppi_lqr")
    fig, axes = plt.subplots(
        1, len(evaders), figsize=(FULL_WIDTH, 2.6), sharey=True, layout="constrained"
    )
    for index, (ax, evader) in enumerate(zip(axes, evaders, strict=True)):
        sub = d[d.evader == evader]
        for sensor, noise, label, line, marker, color in levels:
            cell = sub[sub.sensor == sensor]
            if noise is not None:
                cell = cell[np.isclose(cell.noise_fraction, noise)]
            cell = cell.sort_values("lookahead_s")
            ax.plot(
                cell.lookahead_s,
                100.0 * cell.outcome,
                linestyle=line,
                marker=marker,
                color=color,
                markersize=4.5,
                label=label,
            )
        ax.set_xlabel("MPPI lookahead (s)")
        ax.set_xticks([90, 210, 360, 600, 900])
        ax.set_ylim(-3.0, 103.0)
        ax.text(
            0.03,
            0.96,
            f"{LABELS[evader]} evader",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        letter(ax, index)
    axes[0].set_ylabel("Capture fraction (%)")
    axes[1].legend(loc="upper right", fontsize=8, frameon=True)
    save(fig, "figure_7_lookahead")


def table_6_search():
    """Capture fraction, in percent, of three pursuers against each evader.

    Each entry averages the 25 configurations of the subset of ratios on which
    MCTS was evaluated.
    """
    search = pe_ratios(load("pe_8"))
    subset = search[["thrust_ratio", "budget_ratio"]].drop_duplicates()
    capability = pe_ratios(load("pe_1")).merge(subset, on=["thrust_ratio", "budget_ratio"])
    d = pd.concat([search, capability])
    pursuers = ("hcw", "mppi_flee", "mcts")
    evaders = ("coast", "random", "flee", "mppi_direct", "mppi_lqr", "mcts")
    table = (
        d[d.pursuer.isin(pursuers) & d.evader.isin(evaders)]
        .groupby(["evader", "pursuer"])
        .outcome.mean()
        .unstack()
        .loc[list(evaders), list(pursuers)]
    )
    table = (100.0 * table).round().astype(int)
    table.index = [LABELS[e] for e in table.index]
    table.columns = [LABELS[p] for p in table.columns]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_6_search.txt").write_text(table.to_string() + "\n")
    print(table.to_string())


FIGURES = {
    f.__name__: f
    for f in (
        figure_3_capability,
        figure_4_lbg_capability,
        figure_5_geometry,
        figure_6_information,
        figure_7_lookahead,
        table_6_search,
    )
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "names", nargs="*", default=list(FIGURES), help=f"from: {', '.join(FIGURES)}"
    )
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(FIGURES))
    if unknown:
        parser.error(f"unknown figure {', '.join(unknown)}")
    style()
    for name in args.names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
