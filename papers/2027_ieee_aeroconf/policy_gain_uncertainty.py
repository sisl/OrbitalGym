# /// script
# requires-python = ">=3.12"
# dependencies = ["orbitalgym", "pandas>=2.2", "pyarrow>=17"]
#
# [tool.uv.sources]
# orbitalgym = { path = "../..", editable = true }
# ///
"""Sampling uncertainty of the win fractions and policy gains of the capability experiments.

    uv run papers/2027_ieee_aeroconf/policy_gain_uncertainty.py

Reads the encounter records under ``results/``, so the capability runs of
``pe_1_capability.py`` and ``lbg_1_capability.py`` must have been executed.

In each capability configuration the policy gain is the difference between the
best MPPI policy and the best feedback-control policy, each chosen by its mean
over the encounters. Every policy sees the same initial conditions, so the
gain is a mean of paired per-encounter differences and its standard error
follows from their spread.
"""

import lbg_1_capability
import numpy as np
import pandas as pd
import pe_1_capability
from baseline import FEEDBACK, MPPI_GUARDS, MPPI_PURSUERS
from runner import encounters

PE = dict(
    side="pursuer",
    opponent="evader",
    mppi=MPPI_PURSUERS,
    keys=["evader_cap_mps", "evader_budget_mps"],
)
LBG = dict(
    side="guard",
    opponent="bandit",
    mppi=MPPI_GUARDS,
    keys=["bandit_cap_mps", "bandit_budget_mps"],
)


def paired_gains(records, side, opponent, mppi, keys):
    """Per configuration: the gain, its standard error, and the best policy of each group."""
    rows = []
    for (who, *config), cell in records.groupby([opponent, *keys]):
        table = cell.pivot(index="episode", columns=side, values="outcome")
        best_mppi = table[list(mppi)].mean().idxmax()
        best_feedback = table[list(FEEDBACK)].mean().idxmax()
        diff = table[best_mppi] - table[best_feedback]
        rows.append(
            {
                opponent: who,
                **dict(zip(keys, config, strict=True)),
                "gain": diff.mean(),
                "se": diff.std(ddof=1) / np.sqrt(len(diff)),
            }
        )
    return pd.DataFrame(rows)


def report(run, side, opponent, mppi, keys):
    records = encounters(run)
    for k in keys:
        records[k] = records[k].fillna(np.inf)
    n = run.episodes
    rates = records.groupby([side, opponent, *keys]).outcome.mean()
    se = np.sqrt(rates * (1 - rates) / n)
    print(f"## {run.name}: {n} encounters per configuration, {len(rates)} configurations")
    print(
        f"standard error of a win fraction: largest {se.max():.3f} "
        f"(bound {0.5 / np.sqrt(n):.3f}), mean {se.mean():.3f}"
    )
    gains = paired_gains(records, side, opponent, mppi, keys)
    for who, g in gains.groupby(opponent):
        large = g[g.gain.abs() >= 0.1]
        nonzero = large[(large.gain.abs() - 1.96 * large.se) > 0]
        print(
            f"{who:12s} |gain| >= 0.1 in {len(large):2d} of {len(g)}; of these {len(nonzero):2d} "
            f"differ from zero at 95%; gain standard error: median {g.se.median():.3f}, "
            f"largest {g.se.max():.3f}"
        )


def main():
    runs = {run.name: run for run in (*pe_1_capability.RUNS, *lbg_1_capability.RUNS)}
    for name in ("pe_1", "pe_1_n512"):
        report(runs[name], **PE)
    for name in ("lbg_1", "lbg_1_n512"):
        report(runs[name], **LBG)


if __name__ == "__main__":
    main()
