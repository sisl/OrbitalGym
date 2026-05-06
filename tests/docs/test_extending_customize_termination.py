"""Customize termination — source-of-truth for snippets in
docs/extending/customize-termination.md.
"""

from __future__ import annotations

import jax


def test_customize_termination_walkthrough():
    # --8<-- [start:imports]
    import dataclasses
    from dataclasses import dataclass

    import jax.numpy as jnp

    from orbital_game import OrbitalGameEnv, SingleAgentView, make_lady_bandit_guard
    from orbital_game.termination.reference import MaxStepsOnly
    # --8<-- [end:imports]

    # --8<-- [start:max-steps-only]
    # The framework already ships `MaxStepsOnly` (reads `params.max_steps`
    # at call time, no stored field). Wire it onto the cfg to drop LBG's
    # event-driven termination and keep only the step-cap.
    cfg = make_lady_bandit_guard(seed=0, max_horizon_s=200.0)
    cfg = dataclasses.replace(cfg, termination_fn=MaxStepsOnly())
    # --8<-- [end:max-steps-only]

    # --8<-- [start:and-composite]
    @dataclass(frozen=True)
    class AndComposite:
        """Episode ends only when BOTH conditions are met (rare; usually OR)."""

        first: object
        second: object

        def __call__(self, state, params, t):
            return jnp.logical_and(self.first(state, params, t), self.second(state, params, t))

    # --8<-- [end:and-composite]

    # --8<-- [start:wire-it-up]
    cfg = dataclasses.replace(
        cfg,
        termination_fn=AndComposite(MaxStepsOnly(), MaxStepsOnly()),
    )
    env = OrbitalGameEnv(cfg)
    # --8<-- [end:wire-it-up]

    view = SingleAgentView(env)
    state, obs, opp_ps = view.reset(jax.random.PRNGKey(0))
    del obs
    controlled_cmd = env.guard_command_cls.zeros(cfg.n_guards)
    next_state, next_obs, reward, done, next_opp_ps, info = view.step(
        jax.random.PRNGKey(1), state, controlled_cmd, opp_ps
    )
    del next_state, next_obs, reward, next_opp_ps, info
    composite_done = cfg.termination_fn(state, cfg, state.t)
    assert composite_done.shape == ()
    assert done.shape == ()
