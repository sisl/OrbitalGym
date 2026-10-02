# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""Controlled tests behind the mechanisms that the Results section states.

Each test changes one factor of a stated configuration, simulates 128
encounters, and saves its records to ``summary/diagnostics/<name>.json``.

    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/diagnostics.py
    uv run --with "jax[cuda12]" papers/2027_ieee_aeroconf/diagnostics.py guard_budget drift

Tests:
    budget         delta-v each pursuer needs from 1 km against a coasting evader
    sensing        which part of radial sensing removes the MPPI advantage at
                   the contested operating point
    thrust_race    MPPI against feedback control when budgets never run out
    capture_times  capture-time distributions from 100 m for four horizons
    drift          capture by natural relative motion alone
    guard_budget   whether a larger guard budget lets the feedback-control
                   guards stop the LQR bandit
"""

import argparse
import json
import math
from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
from lbg_game import LBGGame, evaluate_lbg
from pe_game import DEFAULT_PLANNER, Game, Planner, evaluate_games, initial_bank
from runner import SUMMARY

import orbitalgym

EPISODES = 128
SEED = 20260929
CONTESTED = dict(
    evader="flee",
    separation_m=300.0,
    horizon_s=1800.0,
    pursuer_cap_mps=0.1,
    evader_cap_mps=0.1,
    pursuer_budget_mps=5.0,
    evader_budget_mps=3.75,
)
RADIAL = dict(sensor="radial", sensor_range_m=5000.0, prior_sigma_m=30.0, prior_sigma_mps=0.01)


def measure(game, planner=DEFAULT_PLANNER):
    """Capture fraction, delta-v, final range, and capture times of one configuration."""
    bank = initial_bank(EPISODES, SEED, game.separation_m)
    r = evaluate_games([game] * EPISODES, bank, np.arange(EPISODES), planner=planner, seed=SEED)
    times = r["capture_time_s"][np.isfinite(r["capture_time_s"])]
    final_range = np.linalg.norm(r["final_relative"][:, :3], axis=1)
    return {
        "game": {
            k: (str(v) if isinstance(v, float) and math.isinf(v) else v)
            for k, v in asdict(game).items()
        },
        "planner": planner.record(),
        "capture_fraction": float(r["outcome"].mean()),
        "pursuer_dv_mps": float(r["pursuer_dv"].mean()),
        "evader_dv_mps": float(r["evader_dv"].mean()),
        "median_final_range_m": float(np.median(final_range)),
        "capture_time_percentiles_s": (
            np.percentile(times, [10, 50, 90]).tolist() if len(times) else None
        ),
    }


def budget():
    """Delta-v each pursuer needs from 1 km against a coasting evader."""
    return [
        measure(
            Game(
                pursuer=p,
                evader="coast",
                separation_m=1000.0,
                horizon_s=1800.0,
                pursuer_budget_mps=b,
                evader_budget_mps=5.0,
            )
        )
        for b in (5.0, 10.0, math.inf)
        for p in ("lqr", "hcw", "mppi_coast")
    ]


def sensing():
    """Which part of radial sensing removes the MPPI advantage at the contested point."""
    variants = {
        "full": {},
        "both sides, 1% noise": RADIAL,
        "both sides, 0.01% noise": dict(RADIAL, noise_fraction=1e-4),
        "pursuer only, 1% noise": dict(RADIAL, sensed="pursuer"),
        "evader only, 1% noise": dict(RADIAL, sensed="evader"),
        "both sides, 1% noise, no prior error": dict(
            RADIAL, prior_sigma_m=0.0, prior_sigma_mps=0.0
        ),
    }
    return [
        dict(measure(Game(pursuer=p, **CONTESTED, **v)), variant=name)
        for name, v in variants.items()
        for p in ("mppi_flee", "hcw", "lqr")
    ]


def thrust_race():
    """MPPI against feedback when budgets never run out and the thrust margin is 10%."""
    base = dict(
        evader="flee",
        separation_m=300.0,
        horizon_s=1800.0,
        pursuer_cap_mps=0.1,
        evader_cap_mps=0.09,
        pursuer_budget_mps=20.0,
        evader_budget_mps=20.0,
    )
    cases = [
        ("lqr", Planner()),
        ("hcw", Planner()),
        ("mppi_flee", Planner()),
        ("mppi_flee", Planner(noise=0.2)),
        ("mppi_flee", Planner(noise=0.1)),
        ("mppi_flee", Planner(samples=1024)),
    ]
    return [measure(Game(pursuer=p, **base), planner) for p, planner in cases]


def capture_times():
    """Capture-time distributions from 100 m at the contested point for four horizons."""
    return [
        measure(
            Game(
                pursuer=p,
                evader="flee",
                separation_m=100.0,
                horizon_s=t,
                pursuer_budget_mps=5.0,
                evader_budget_mps=3.75,
            )
        )
        for t in (400.0, 600.0, 900.0, 1800.0)
        for p in ("direct", "lqr", "hcw", "mppi_flee", "mppi_coast")
    ]


def drift():
    """Capture by natural relative motion alone: both sides coast from a 300 m offset."""
    records = []
    for axis, name in ((0, "radial"), (1, "along-track"), (2, "cross-track")):
        initial = np.zeros((1, 6))
        initial[0, axis] = 300.0
        game = Game(
            pursuer="coast",
            evader="coast",
            separation_m=300.0,
            horizon_s=1800.0,
            pursuer_budget_mps=0.0,
            evader_budget_mps=0.0,
        )
        r = evaluate_games([game], initial, [0], seed=SEED)
        records.append(
            {
                "offset_axis": name,
                "captured": int(r["outcome"][0]),
                "capture_time_s": float(r["capture_time_s"][0]),
                "closest_m": float(r["closest_m"][0]),
            }
        )
    return records


def guard_budget():
    """Whether a larger guard budget lets the feedback-control guards stop the LQR bandit.

    Guard and bandit have equal thrust limits and the bandit keeps a 5 m/s budget.
    """
    records = []
    ids = np.arange(EPISODES)
    for guard in ("lqr", "hcw", "pd"):
        for limit in (5.0, 10.0, 20.0, math.inf):
            game = LBGGame(guard=guard, bandit="lqr", guard_budget_mps=limit, bandit_budget_mps=5.0)
            r = evaluate_lbg([game] * EPISODES, game.initial_states(EPISODES, SEED), ids, seed=SEED)
            records.append(
                {
                    "guard": guard,
                    "bandit": "lqr",
                    "guard_budget_mps": str(limit) if math.isinf(limit) else limit,
                    "defense_fraction": float(np.mean(r["outcome"])),
                    "capture_fraction": float(np.mean(r["capture"])),
                    "breach_fraction": float(np.mean(r["breach"])),
                    "guard_dv_mps": float(np.mean(r["guard_dv"])),
                    "median_closest_m": float(np.median(r["closest_m"])),
                }
            )
    return records


TESTS = {f.__name__: f for f in (budget, sensing, thrust_race, capture_times, drift, guard_budget)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("names", nargs="*", default=list(TESTS), help=f"from: {', '.join(TESTS)}")
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(TESTS))
    if unknown:
        parser.error(f"unknown test {', '.join(unknown)}")
    orbitalgym.set_precision(jnp.float64)
    out = SUMMARY / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        records = TESTS[name]()
        (out / f"{name}.json").write_text(json.dumps(records, indent=2) + "\n")
        print(name, len(records), flush=True)


if __name__ == "__main__":
    main()
