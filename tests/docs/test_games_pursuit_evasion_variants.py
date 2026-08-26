"""Pursuit-Evasion variants — source-of-truth for the Variants subsection in
docs/games/pursuit-evasion.md.
"""

from __future__ import annotations


def test_pursuit_evasion_variants():
    # --8<-- [start:variant-jittered-pursuer]
    import dataclasses

    from orbitalgym import make_pursuit_evasion
    from orbitalgym.policies.heuristic import JitteredPolicy, LeadInterceptPursuer

    cfg = make_pursuit_evasion(seed=0, max_horizon_s=200.0)
    base = LeadInterceptPursuer(max_dv_mps=0.05, dt=cfg.dt)
    cfg = dataclasses.replace(cfg, bandit_policy=JitteredPolicy(base=base, sigma=0.005))
    # --8<-- [end:variant-jittered-pursuer]

    # --8<-- [start:variant-tighter-ic]
    import jax.numpy as jnp

    from orbitalgym.sampling.side import RelativeEllipse
    from orbitalgym.sampling.spec import ICSpec

    tight_ic = ICSpec(
        guard_sampler=RelativeEllipse(
            radial_ellipse_m=1000.0,
            cross_track_m=0.0,
            along_track_offset_m=0.0,
            phase_rad=0.0,
            sigma_radial_ellipse_m=1.0,
        ),
        bandit_sampler=RelativeEllipse(
            radial_ellipse_m=500.0,
            cross_track_m=100.0,
            along_track_offset_m=200.0,
            phase_rad=jnp.pi / 2.0,
            sigma_radial_ellipse_m=1.0,
        ),
        validators=(),
        max_attempts=100,
    )
    cfg2 = dataclasses.replace(cfg, ic_sampler=tight_ic)
    # --8<-- [end:variant-tighter-ic]

    assert cfg.bandit_policy is not None
    assert isinstance(cfg.bandit_policy, JitteredPolicy)
    assert cfg2.ic_sampler.max_attempts == 100
