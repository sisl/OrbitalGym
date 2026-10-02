# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10", "numpy", "pandas>=2.2", "scienceplots>=2.2"]
# ///
"""Print the quantities that the Results section quotes, from the committed summaries.

    uv run papers/2027_ieee_aeroconf/quoted_numbers.py

Fractions are printed as decimals; the paper reports them as percentages. The
policy gain of a configuration is the win fraction of the best MPPI policy
minus that of the best feedback-control policy.
"""

import json

import numpy as np
import pandas as pd
from figures import (
    FEEDBACK,
    MPPI_GUARDS,
    MPPI_PURSUERS,
    SUMMARY,
    lbg_ratios,
    load,
    pe_ratios,
)

RATIOS = ["thrust_ratio", "budget_ratio"]
CONTESTED = dict(thrust_ratio=1.0, budget_ratio=0.75)


def best(frame, policies, keys, column="pursuer"):
    """Win fraction of the best policy of a group in each configuration."""
    return frame[frame[column].isin(policies)].groupby(keys).outcome.max()


def gain(frame, keys, mppi=MPPI_PURSUERS, column="pursuer"):
    return best(frame, mppi, keys, column) - best(frame, FEEDBACK, keys, column)


def at(frame, **match):
    """Rows whose columns equal the given values."""
    for key, value in match.items():
        column = frame[key]
        frame = frame[np.isclose(column, value) if isinstance(value, float) else column == value]
    return frame


def meaningful(gains):
    return int((gains.abs() >= 0.1).sum())


def capability():
    print("## PE-1 and PE-2: capability and budget size")
    d = pe_ratios(pd.concat([load("pe_1"), load("pe_2")]))
    five = d[d.pursuer_budget_mps == 5.0]
    for evader, sub in five.groupby("evader"):
        g = gain(sub, RATIOS)
        large = g[g.abs() >= 0.1]
        print(
            f"{evader}: |gain| >= 0.1 in {len(large)} of {len(g)}; by budget ratio "
            f"{large.groupby(level='budget_ratio').size().to_dict()}; largest {g.max():.2f} at "
            f"{g.idxmax()}, smallest {g.min():.2f} at {g.idxmin()}"
        )
    flee = d[d.evader == "flee"]
    for budget in (1.0, 5.0, 20.0):
        sub = flee[flee.pursuer_budget_mps == budget]
        feedback, mppi = best(sub, FEEDBACK, RATIOS), best(sub, MPPI_PURSUERS, RATIOS)
        quarter = mppi.xs(0.25, level="budget_ratio")
        both = np.maximum(feedback, mppi)
        print(
            f"budget {budget:g} m/s, fleeing evader: feedback largest {feedback.max():.3f}; "
            f"MPPI at budget ratio 0.25 {quarter.min():.3f} to {quarter.max():.3f}; "
            f"best pursuer at (1, 0.75) {both.loc[(1.0, 0.75)]:.3f}"
        )
    unlimited = flee[np.isinf(flee.pursuer_budget_mps)]
    print(
        "unlimited budgets, fleeing evader, best pursuer by thrust ratio:",
        best(unlimited, FEEDBACK + MPPI_PURSUERS, ["thrust_ratio"]).round(3).to_dict(),
    )
    for evader in ("coast", "random"):
        every = best(five[five.evader == evader], FEEDBACK + MPPI_PURSUERS, RATIOS)
        print(f"{evader} evader at 5 m/s: best pursuer captures at least {every.min():.3f}")
    here = at(five, evader="flee", thrust_ratio=1.1, budget_ratio=0.75).set_index("pursuer")
    print("fleeing evader at (1.1, 0.75):", here.outcome.round(3).to_dict())
    here = at(five, evader="flee", thrust_ratio=0.75, budget_ratio=0.9).set_index("pursuer")
    print("fleeing evader at (0.75, 0.9):", here.outcome.round(3).to_dict())


def repetitions():
    print("## Independent initial conditions")
    keys = ["pursuer", "evader", *RATIOS]
    base = pe_ratios(load("pe_1"))
    for run in ("pe_1_seed2", "pe_1_n512"):
        other = pe_ratios(load(run))
        diff = (base.set_index(keys).outcome - other.set_index(keys).outcome).abs()
        counts = {evader: meaningful(gain(sub, RATIOS)) for evader, sub in other.groupby("evader")}
        flee = gain(other[other.evader == "flee"], RATIOS)
        print(
            f"{run}: mean |difference| from pe_1 {diff.mean():.4f}; meaningful gains by evader "
            f"{counts}; fleeing evader largest {flee.max():.2f} at {flee.idxmax()}, smallest "
            f"{flee.min():.2f} at {flee.idxmin()}"
        )
    keys = ["guard", "bandit", *RATIOS]
    base, other = lbg_ratios(load("lbg_1")), lbg_ratios(load("lbg_1_n512"))
    diff = (base.set_index(keys).outcome - other.set_index(keys).outcome).abs()
    counts = {
        bandit: meaningful(gain(sub, RATIOS, MPPI_GUARDS, "guard"))
        for bandit, sub in other.groupby("bandit")
    }
    print(f"lbg_1_n512: mean |difference| from lbg_1 {diff.mean():.4f}; meaningful gains {counts}")


def lady_bandit_guard():
    print("## LBG-1 to LBG-3")
    d = lbg_ratios(load("lbg_1"))
    for bandit, sub in d.groupby("bandit"):
        g = gain(sub, RATIOS, MPPI_GUARDS, "guard")
        idle = best(sub, ("coast",), RATIOS, "guard")
        print(
            f"{bandit} bandit: gain >= 0.1 in {(g >= 0.1).sum()} of {len(g)}, <= -0.1 in "
            f"{(g <= -0.1).sum()}; an idle guard defends at least {idle.min():.2f}"
        )
    equal = at(d, thrust_ratio=1.0, budget_ratio=1.0)
    for bandit in ("lqr", "mppi_lqr"):
        sub = equal[equal.bandit == bandit].set_index("guard")
        print(
            f"equal capability, {bandit} bandit: defense {sub.outcome.round(3).to_dict()}; guard "
            f"delta-v {sub.guard_dv.round(2).to_dict()}; closest {sub.closest_m.round(1).to_dict()}"
        )
    double = at(d, budget_ratio=2.0)
    idle = double[double.guard == "coast"].groupby("bandit").outcome.min()
    print(
        "guard with twice the bandit's budget, idle guard, least defense:", idle.round(3).to_dict()
    )
    geometry = load("lbg_2")
    idle = geometry[geometry.guard == "coast"]
    far = idle[idle.bandit_separation_m >= 2000.0].groupby("bandit").outcome.min()
    print("bandit from 2 km or more, idle guard, least defense:", far.round(3).to_dict())
    table = geometry.set_index(["guard", "bandit", "bandit_separation_m", "horizon_s"]).outcome
    table = table.unstack("horizon_s")
    change = (table[[1800.0, 3600.0]].sub(table[1200.0], axis=0)).abs().max().max()
    print(f"largest change in defense beyond a 1,200 s horizon: {change:.3f}")
    events = load("lbg_3")
    keys = ["bandit", "capture_m", "breach_m"]
    for matched, sub in events.groupby(np.isfinite(events.capture_speed_mps)):
        g = gain(sub, keys, MPPI_GUARDS, "guard")
        print(f"velocity matching {matched}: largest MPPI advantage {g.max():.3f}")


def geometry():
    print("## PE-3: geometry")
    d = pe_ratios(load("pe_3"))
    keys = ["evader", *RATIOS, "separation_m", "horizon_s"]
    for name, group in (("feedback", FEEDBACK), ("MPPI", MPPI_PURSUERS)):
        table = best(d, group, keys).unstack("horizon_s")
        change = (table[[3600.0, 5400.0]].max(axis=1) - table[1800.0]).abs().max()
        print(f"{name}: largest change of the best policy beyond 1,800 s: {change:.3f}")
    coast = at(d, evader="coast", horizon_s=1800.0, **CONTESTED)
    for separation in (800.0, 1600.0):
        sub = coast[coast.separation_m == separation].set_index("pursuer")
        print(
            f"coasting evader from {separation:g} m: capture {sub.outcome.round(2).to_dict()}; "
            f"delta-v {sub.pursuer_dv.round(2).to_dict()}; median capture time "
            f"{sub.median_capture_time_s.to_dict()}"
        )
    near = at(d, evader="flee", separation_m=100.0, **CONTESTED)
    table = near.set_index(["pursuer", "horizon_s"]).outcome.unstack()[[600.0, 900.0]]
    print("fleeing evader from 100 m, capture by horizon:")
    print(table.round(2).to_string())


def information():
    print("## PE-4: information")
    d = pe_ratios(load("pe_4"))
    coast = at(d, evader="coast", separation_m=1000.0, **CONTESTED)
    feedback = coast[coast.pursuer.isin(FEEDBACK)].outcome.max()
    print(f"coasting evader at 1 km, feedback largest: {feedback:.3f}")
    columns = ["sensor", "sensor_range_m", "noise_fraction", "outcome"]
    print(coast[coast.pursuer == "mppi_flee"][columns].round(3).to_string(index=False))
    flee = at(d, evader="flee", separation_m=300.0, **CONTESTED)
    radial = at(flee, sensor="radial", sensor_range_m=5000.0, noise_fraction=0.01)
    print(
        "contested point, radial 5 km sensor: capture",
        radial.set_index("pursuer").outcome.round(3).to_dict(),
        "mean estimate error (m)",
        radial.set_index("pursuer").pursuer_mean_error_m.round(2).to_dict(),
    )
    full = flee[flee.sensor == "full"].set_index("pursuer").outcome
    print("contested point, full information:", full.round(3).to_dict())
    sensed, known = pe_ratios(load("pe_4_capability")), pe_ratios(load("pe_1"))
    every = FEEDBACK + MPPI_PURSUERS
    lower = best(sensed, every, ["evader", *RATIOS]) < best(known, every, ["evader", *RATIOS])
    print(f"cone sensor: the best pursuer captures less in {lower.sum()} of {len(lower)}")
    for evader, sub in sensed.groupby("evader"):
        g = gain(sub, RATIOS)
        every = best(sub, FEEDBACK + MPPI_PURSUERS, RATIOS)
        print(
            f"cone sensor, {evader}: |gain| >= 0.1 in {meaningful(g)} of {len(g)} "
            f"(feedback better in {(g <= -0.1).sum()}, MPPI in {(g >= 0.1).sum()}); "
            f"best pursuer captures at least {every.min():.2f}"
        )


def policy_settings():
    print("## PE-5, PE-6, PE-8: policy settings")
    d = load("pe_5")
    d["lookahead_s"] = d.planner_segments * d.planner_repeat * d.dt_s
    table = d.groupby(["evader", "sensor", "noise_fraction", "lookahead_s"]).outcome.mean()
    print(table.unstack().round(3).to_string())
    d = pe_ratios(load("pe_6"))
    keys = ["evader", "thrust_ratio", "pursuer", "planner_evader_samples"]
    table = d.groupby(keys).outcome.mean().unstack()
    print(table.round(3).to_string())
    print(f"largest change from 64 to 1,024 samples: {(table[1024] - table[64]).abs().max():.3f}")
    lookahead = pd.concat([load("pe_8_hcw_lookahead"), at(load("pe_1"), pursuer="hcw")])
    coast = lookahead[lookahead.evader == "coast"].groupby("lookahead_s")
    print(
        "HCW intercept against a coasting evader by lookahead: capture",
        coast.outcome.mean().round(3).to_dict(),
        "delta-v",
        coast.pursuer_dv.mean().round(2).to_dict(),
    )


def capture_definition():
    print("## PE-7: capture definition")
    d = pe_ratios(load("pe_7"))
    flee = at(d, evader="flee", **CONTESTED)
    for speed, sub in flee.groupby("capture_speed_mps"):
        table = pd.DataFrame(
            {
                "feedback": best(sub, FEEDBACK, ["capture_m"]),
                "mppi": best(sub, MPPI_PURSUERS, ["capture_m"]),
            }
        )
        table["gain"] = table.mppi - table.feedback
        print(f"velocity-matching condition {speed} m/s:")
        print(table.round(3).to_string())
    keys = ["pursuer", "evader", *RATIOS]
    change = (
        pe_ratios(load("pe_1")).set_index(keys).outcome
        - pe_ratios(load("pe_7_endpoint")).set_index(keys).outcome
    )
    print("capture tested at the end of each step only, largest |change| by pursuer:")
    print(change.abs().groupby("pursuer").max().round(3).to_string())


def diagnostics():
    print("## Diagnostics")
    for path in sorted((SUMMARY / "diagnostics").glob("*.json")):
        for record in json.loads(path.read_text()):
            game = record.pop("game", {})
            planner = record.pop("planner", {})
            label = {
                k: game[k] for k in ("pursuer", "horizon_s", "pursuer_budget_mps") if k in game
            }
            if planner and (planner["noise"], planner["samples"]) != (0.5, 256):
                label |= {"noise": planner["noise"], "samples": planner["samples"]}
            values = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()}
            print(path.stem, label, values)


def main():
    capability()
    repetitions()
    lady_bandit_guard()
    geometry()
    information()
    policy_settings()
    capture_definition()
    diagnostics()


if __name__ == "__main__":
    main()
