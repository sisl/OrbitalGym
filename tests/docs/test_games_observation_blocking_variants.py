"""Observation-Blocking variants — source-of-truth for the Variants subsection in
docs/games/observation-blocking.md.
"""

from __future__ import annotations


def test_observation_blocking_variants():
    # --8<-- [start:variant-lead-intercept-bandit]
    import dataclasses

    from orbital_game import make_observation_blocking
    from orbital_game.policies.heuristic import LeadInterceptPursuer

    cfg = make_observation_blocking(seed=0, max_horizon_s=5400.0)
    bandit = LeadInterceptPursuer(max_dv_mps=0.05, dt=cfg.dt)
    cfg = dataclasses.replace(cfg, bandit_policy=bandit)
    # --8<-- [end:variant-lead-intercept-bandit]

    # --8<-- [start:variant-tighter-ic]
    import jax.numpy as jnp

    from orbital_game.sampling.side import RelativeEllipse
    from orbital_game.sampling.spec import ICSpec

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

    assert isinstance(cfg.bandit_policy, LeadInterceptPursuer)
    assert cfg2.ic_sampler.max_attempts == 100
