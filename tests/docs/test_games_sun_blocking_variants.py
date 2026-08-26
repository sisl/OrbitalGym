"""Sun-Blocking variants — source-of-truth for the Variants subsection in
docs/games/sun-blocking.md.
"""

from __future__ import annotations


def test_sun_blocking_variants():
    # --8<-- [start:variant-sun-tracker-bandit]
    import dataclasses

    import jax.numpy as jnp

    from orbitalgym import make_sun_blocking
    from orbitalgym.policies.heuristic import SunTrackerBlocker

    cfg = make_sun_blocking(seed=0, max_horizon_s=5400.0)
    sun_bandit = SunTrackerBlocker(
        sun_dir_rtn=jnp.array([1.0, 0.0, 0.0]),
        max_dv_mps=0.05,
    )
    cfg = dataclasses.replace(cfg, bandit_policy=sun_bandit)
    # --8<-- [end:variant-sun-tracker-bandit]

    # --8<-- [start:variant-jittered-sun-tracker]
    from orbitalgym.policies.heuristic import JitteredPolicy

    base = SunTrackerBlocker(
        sun_dir_rtn=jnp.array([1.0, 0.0, 0.0]),
        max_dv_mps=0.05,
    )
    cfg2 = dataclasses.replace(cfg, bandit_policy=JitteredPolicy(base=base, sigma=0.005))
    # --8<-- [end:variant-jittered-sun-tracker]

    assert isinstance(cfg.bandit_policy, SunTrackerBlocker)
    assert isinstance(cfg2.bandit_policy, JitteredPolicy)
