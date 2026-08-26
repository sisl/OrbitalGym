"""Heuristic policy cookbook — source-of-truth for snippets in
docs/extending/heuristic-policy-cookbook.md.
"""

from __future__ import annotations


def test_heuristic_cookbook_walkthrough():
    # --8<-- [start:imports]
    import dataclasses

    from orbitalgym import make_pursuit_evasion
    from orbitalgym.policies.heuristic import (
        JitteredPolicy,
        LeadInterceptPursuer,
        OrthogonalEvader,
    )
    # --8<-- [end:imports]

    # --8<-- [start:lead-intercept-wire]
    cfg = make_pursuit_evasion(seed=0, max_horizon_s=200.0)
    cfg = dataclasses.replace(cfg, bandit_policy=LeadInterceptPursuer(max_dv_mps=0.05, dt=cfg.dt))
    # --8<-- [end:lead-intercept-wire]

    # --8<-- [start:randomized-jitter-wire]
    base = LeadInterceptPursuer(max_dv_mps=0.05, dt=cfg.dt)
    cfg2 = dataclasses.replace(cfg, bandit_policy=JitteredPolicy(base=base, sigma=0.005))
    # --8<-- [end:randomized-jitter-wire]

    # --8<-- [start:evader-wire]
    cfg3 = make_pursuit_evasion(seed=0, max_horizon_s=200.0)
    cfg3 = dataclasses.replace(cfg3, bandit_policy=OrthogonalEvader(max_dv_mps=0.05))
    # --8<-- [end:evader-wire]

    # Smoke checks — all three cfgs constructed successfully.
    assert cfg.bandit_policy is not None
    assert cfg2.bandit_policy is not None
    assert cfg3.bandit_policy is not None
