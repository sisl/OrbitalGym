"""Lady-Bandit-Guard variants — source-of-truth for the Variants subsection in
docs/games/lady-bandit-guard.md.
"""

from __future__ import annotations


def test_lady_bandit_guard_variants():
    # --8<-- [start:variant-jittered-bandit]
    import dataclasses

    from orbital_game import make_lady_bandit_guard
    from orbital_game.policies.heuristic import JitteredPolicy, LeadInterceptPursuer

    cfg = make_lady_bandit_guard(seed=0, max_horizon_s=2000.0)
    base = LeadInterceptPursuer(max_dv_mps=0.05, dt=cfg.dt)
    cfg = dataclasses.replace(cfg, bandit_policy=JitteredPolicy(base=base, sigma=0.005))
    # --8<-- [end:variant-jittered-bandit]

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
            radial_ellipse_m=1000.0,
            cross_track_m=0.0,
            along_track_offset_m=0.0,
            phase_rad=jnp.pi,
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
