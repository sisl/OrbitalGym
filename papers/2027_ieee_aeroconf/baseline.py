"""Baseline scenario shared by the experiments, and the capability sweep several of them reuse.

The pursuing side, the pursuer in pursuit-evasion and the guard in
lady-bandit-guard, has a thrust limit of 0.1 m/s per 10 s step and a delta-v
budget of 5 m/s. Ratios set the other side's limits. In pursuit-evasion they
divide the evader's capability by the pursuer's, which is the guard-over-bandit
ratio of the paper. In lady-bandit-guard the grids are built from the
bandit-over-guard ratio, the reciprocal of the paper's.
"""

import itertools

import numpy as np
from lbg_game import LBGGame
from pe_game import Game

THRUST_LIMIT = 0.1
BUDGET = 5.0
RATIOS = (0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
BUDGET_RATIOS = (*RATIOS, np.inf)
FINE_RATIOS = tuple(float(v) for v in np.round(2.0 ** np.linspace(-2, 1, 31), 6))

PURSUERS = ("coast", "direct", "pd", "lqr", "hcw", "mppi_coast", "mppi_flee")
EVADERS = ("coast", "random", "flee", "transverse", "mppi_direct", "mppi_lqr")
FOCUS_EVADERS = ("coast", "random", "flee", "mppi_lqr")
# Policy groups compared in the Results section.
FEEDBACK = ("direct", "pd", "lqr", "hcw")
MPPI_PURSUERS = ("mppi_coast", "mppi_flee")
MPPI_GUARDS = ("mppi_coast", "mppi_approach")
GROUP_PURSUERS = FEEDBACK + MPPI_PURSUERS

# (thrust ratio, budget ratio): contested, thrust advantage, budget advantage.
OPERATING_POINTS = ((1.0, 0.75), (0.5, 1.0), (1.0, 0.5))


def pe_game(thrust_ratio, budget_ratio, budget=BUDGET, **settings):
    """Pursuit-evasion from 300 m over 1,800 s at the given evader-over-pursuer ratios."""
    return Game(
        pursuer_cap_mps=THRUST_LIMIT,
        evader_cap_mps=THRUST_LIMIT * thrust_ratio,
        pursuer_budget_mps=budget,
        evader_budget_mps=budget * budget_ratio,
        **{"separation_m": 300.0, "horizon_s": 1800.0, **settings},
    )


def capability_grid(budget=BUDGET, **settings):
    """Every pursuer against every evader over the thrust ratio and the budget ratio."""
    return [
        pe_game(thrust, fuel, budget, pursuer=p, evader=e, **settings)
        for p, e, thrust, fuel in itertools.product(PURSUERS, EVADERS, RATIOS, BUDGET_RATIOS)
    ]


def fine_capability_grid(budgets):
    """Both policy groups against the fleeing evader on a 31 by 31 grid of ratios."""
    return [
        pe_game(thrust, fuel, budget, pursuer=p, evader="flee")
        for budget, p, thrust, fuel in itertools.product(
            budgets, GROUP_PURSUERS, FINE_RATIOS, FINE_RATIOS
        )
    ]


def lbg_game(thrust_ratio, budget_ratio, **settings):
    """Lady-bandit-guard at the given bandit-over-guard ratios."""
    return LBGGame(
        guard_cap_mps=THRUST_LIMIT,
        bandit_cap_mps=THRUST_LIMIT * thrust_ratio,
        guard_budget_mps=BUDGET,
        bandit_budget_mps=BUDGET * budget_ratio,
        **settings,
    )
