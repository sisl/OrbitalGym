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
    # --8<-- [end:imports]

    # --8<-- [start:max-steps-only]
    @dataclass(frozen=True)
    class MaxStepsOnly:
        """Episode ends only when step count hits max_steps."""

        max_steps: int

        def __call__(self, state, params, t):
            del params, t
            return state.step >= self.max_steps

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
    cfg = make_lady_bandit_guard(seed=0, max_horizon_s=200.0)
    cfg = dataclasses.replace(cfg, termination_fn=MaxStepsOnly(max_steps=cfg.max_steps))
    env = OrbitalGameEnv(cfg)
    # --8<-- [end:wire-it-up]

    view = SingleAgentView(env)
    state, obs, opp_ps = view.reset(jax.random.PRNGKey(0))
    del obs
    next_state, next_obs, reward, done, next_opp_ps, info = view.step(
        jax.random.PRNGKey(1), state, jnp.zeros((cfg.n_guards, 3)), opp_ps
    )
    del next_state, next_obs, reward, next_opp_ps, info
    # `AndComposite` is exercised here to keep ruff F841 quiet and to smoke-test
    # the composite shape — combine MaxStepsOnly with itself.
    composite = AndComposite(
        MaxStepsOnly(max_steps=cfg.max_steps),
        MaxStepsOnly(max_steps=cfg.max_steps),
    )
    composite_done = composite(state, cfg, state.t)
    assert composite_done.shape == ()
    assert done.shape == ()
