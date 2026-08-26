"""Customize IC sampling — source-of-truth for snippets in
docs/extending/customize-ic-sampling.md.
"""

from __future__ import annotations

import jax


def test_customize_ic_sampling_walkthrough():
    # --8<-- [start:imports]
    import dataclasses

    import jax.numpy as jnp

    from orbitalgym import OrbitalGymEnv, SingleAgentView, make_pursuit_evasion
    from orbitalgym.sampling.side import RelativeEllipse
    from orbitalgym.sampling.spec import ICSpec
    # --8<-- [end:imports]

    # --8<-- [start:tighter-gaussian]
    tight_ic = ICSpec(
        guard_sampler=RelativeEllipse(
            radial_ellipse_m=1000.0,
            cross_track_m=0.0,
            along_track_offset_m=0.0,
            phase_rad=0.0,
            sigma_radial_ellipse_m=1.0,  # 10x tighter than default
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
    # --8<-- [end:tighter-gaussian]

    # --8<-- [start:wire-it-up]
    cfg = make_pursuit_evasion(seed=0, max_horizon_s=200.0)
    cfg = dataclasses.replace(cfg, ic_sampler=tight_ic)
    env = OrbitalGymEnv(cfg)
    # --8<-- [end:wire-it-up]

    view = SingleAgentView(env)
    state, obs, opp_ps = view.reset(jax.random.PRNGKey(0))
    assert state.guards.rtn.shape == (cfg.n_guards, 6)
