"""Run the configurations of an experiment, record every encounter, and summarize.

A run is a named list of configurations. Every configuration is simulated on
the same initial conditions and seeds, so two configurations are compared on
the same encounters. A run writes one Parquet file per configuration under
``results/<run>/cells`` and resumes from the files already there. When it
finishes it writes ``summary/<run>.csv`` with one row per configuration, which
is what the figures and the quoted statistics read.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from lbg_game import LBGGame, evaluate_lbg
from pe_game import Planner, evaluate_games

import orbitalgym

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY = ROOT / "summary"
SHARED_SOURCES = ("controllers.py", "pe_game.py", "lbg_game.py", "baseline.py", "runner.py")
MEANS = (
    "pursuer_dv",
    "evader_dv",
    "guard_dv",
    "bandit_dv",
    "capture",
    "breach",
    "closest_m",
    "final_range_m",
    "pursuer_detection_fraction",
    "pursuer_mean_error_m",
)
TIMES = ("capture_time_s", "event_time_s")
LIBRARIES = ("jax", "jaxlib", "numpy", "scipy", "mctx")


@dataclass(frozen=True)
class Run:
    """A named grid of configurations with its number of encounters and seed.

    ``grid`` returns games, or (game, planner) pairs where the MPPI settings
    differ from the defaults.
    """

    name: str
    grid: Callable[[], list]
    episodes: int = 128
    seed: int = 20260929
    batch_cells: int = 90

    def entries(self):
        """Configurations as (game, planner) pairs."""
        return [e if isinstance(e, tuple) else (e, Planner()) for e in self.grid()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def identity(run):
    """What must not change between the launch of a run and its resumption."""
    sources = [ROOT / name for name in SHARED_SOURCES]
    sources.append(Path(inspect.getsourcefile(run.grid)).resolve())
    return {
        "run": run.name,
        "episodes": run.episodes,
        "seed": run.seed,
        "default_planner": Planner().record(),
        "orbitalgym": importlib.metadata.version("orbitalgym"),
        "sources": {p.name: digest(p) for p in sorted(set(sources))},
        "initial_population": "isotropic position direction, zero relative velocity",
    }


def batches(pairs, size):
    """Groups of configurations that one compiled program can evaluate together."""
    groups = {}
    for game, planner in pairs:
        groups.setdefault((planner, game.horizon_s, game.batch_signature), []).append(
            (game, planner)
        )
    for _, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for start in range(0, len(members), size):
            yield members[start : start + size]


def cell_frame(game, planner, result, part, ids):
    """Encounter records of one configuration, with its settings as columns."""
    final = np.asarray(result["final_relative"])[part, :3]
    frame = pd.DataFrame(
        {
            "episode": ids,
            **{k: np.asarray(v)[part] for k, v in result.items() if k != "final_relative"},
            "final_range_m": np.linalg.norm(final, axis=1),
        }
    )
    for name, value in asdict(game).items():
        frame[name] = value
    for name, value in asdict(planner).items():
        frame[f"planner_{name}"] = value
    frame["planner_evader_samples"] = planner.for_evader().samples
    frame["planner_evader_segments"] = planner.for_evader().segments
    return frame


def execute(run, results=RESULTS):
    """Simulate every configuration of the run that has no saved record yet."""
    orbitalgym.set_precision(jnp.float64)
    out = Path(results) / run.name
    (out / "cells").mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    invocation = {
        "utc": now,
        "command": sys.argv,
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "versions": {name: importlib.metadata.version(name) for name in LIBRARIES},
    }
    path = out / "run.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["identity"] != identity(run):
            raise RuntimeError(
                f"{out} holds a run with a different grid, seed, encounter count, planner, "
                "or source; move it aside or pass another --results directory"
            )
        manifest["invocations"].append(invocation)
        manifest.pop("finished_utc", None)
    else:
        manifest = {"identity": identity(run), "started_utc": now, "invocations": [invocation]}
    atomic_json(path, manifest)
    pairs = run.entries()
    todo = [(g, p) for g, p in pairs if not (out / "cells" / f"{g.key(p)}.parquet").exists()]
    print(
        f"{run.name}: {len(pairs)} configurations, {len(todo)} remaining, "
        f"backend {jax.default_backend()}",
        flush=True,
    )
    ids = np.arange(run.episodes)
    for batch in batches(todo, run.batch_cells):
        start = time.time()
        planner = batch[0][1]
        rows = [(game, game.initial_states(run.episodes, run.seed)) for game, _ in batch]
        evaluate = evaluate_lbg if isinstance(batch[0][0], LBGGame) else evaluate_games
        result = evaluate(
            [g for g, bank in rows for _ in ids],
            np.concatenate([bank for _, bank in rows]),
            np.tile(ids, len(rows)),
            planner=planner,
            seed=run.seed,
        )
        for i, (game, _) in enumerate(rows):
            part = slice(i * len(ids), (i + 1) * len(ids))
            frame = cell_frame(game, planner, result, part, ids)
            frame.to_parquet(out / "cells" / f"{game.key(planner)}.parquet")
        print(
            f"batch {len(batch)} configurations plans={batch[0][0].plans} "
            f"{time.time() - start:.1f}s",
            flush=True,
        )
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    atomic_json(path, manifest)


def encounters(run, results=RESULTS):
    """Encounter records of every configuration of a finished run, checked for completeness."""
    out = Path(results) / run.name
    if "finished_utc" not in json.loads((out / "run.json").read_text()):
        raise ValueError(f"{run.name}: the run has not finished")
    expected = np.arange(run.episodes)
    frames = []
    for game, planner in run.entries():
        path = out / "cells" / f"{game.key(planner)}.parquet"
        if not path.exists():
            raise ValueError(f"{run.name}: missing configuration {game.key(planner)}")
        frame = pd.read_parquet(path)
        if not np.array_equal(np.sort(frame.episode.values), expected):
            raise ValueError(f"{run.name}: incomplete encounters in {path.name}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(run, results=RESULTS, summary=SUMMARY):
    """Reduce the encounter records to one row per configuration.

    ``outcome`` is the fraction of encounters the pursuer (pursuit-evasion) or
    the guard (lady-bandit-guard) wins.
    """
    frame = encounters(run, results)
    measured = ("episode", "outcome", "duration_s", *MEANS, *TIMES)
    config = [c for c in frame.columns if c not in measured]
    for column in config:
        if frame[column].dtype.kind == "f":
            frame[column] = frame[column].fillna(np.inf)
    grouped = frame.groupby(config, dropna=False)
    table = grouped.agg(
        episodes=("outcome", "size"),
        wins=("outcome", "sum"),
        **{c: (c, "mean") for c in MEANS if c in frame},
    )
    table["outcome"] = table.wins / table.episodes
    for column in TIMES:
        if column in frame:
            table[f"median_{column}"] = grouped[column].apply(
                lambda t: np.median(t[np.isfinite(t)]) if np.isfinite(t).any() else np.nan
            )
    Path(summary).mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(Path(summary) / f"{run.name}.csv", index=False)
    return len(table)


def main(runs, description):
    """Command line of an experiment script: execute and summarize the named runs."""
    by_name = {run.name: run for run in runs}
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs",
        nargs="*",
        default=list(by_name),
        metavar="run",
        help=f"runs to execute, from: {', '.join(by_name)} (default: all)",
    )
    parser.add_argument("--results", default=RESULTS, help="directory of encounter records")
    parser.add_argument("--summary", default=SUMMARY, help="directory of summary tables")
    args = parser.parse_args()
    unknown = sorted(set(args.runs) - set(by_name))
    if unknown:
        parser.error(f"unknown run {', '.join(unknown)}; choose from {', '.join(by_name)}")
    for name in args.runs:
        execute(by_name[name], args.results)
        rows = summarize(by_name[name], args.results, args.summary)
        print(f"{name}: summarized {rows} configurations", flush=True)
