"""Tests for the symmetric core's step semantics."""

from __future__ import annotations

import jax

from examples.reference_scenario import build_config
from orbital_game.env.core import OrbitalGameEnv
from orbital_game.env.types import Actions, BySide, SideOutput, StepOutput


def _zero_actions(env: OrbitalGameEnv) -> Actions:
    cfg = env.config
    return Actions(
        sides=BySide(
            guard=env.guard_command_cls.zeros(cfg.n_guards),
            bandit=env.bandit_command_cls.zeros(cfg.n_bandits),
        )
    )


def test_step_returns_step_output():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))
    actions = _zero_actions(env)
    out = env.step(jax.random.PRNGKey(1), state, actions)
    assert isinstance(out, StepOutput)
    assert isinstance(out.outputs, BySide)
    assert isinstance(out.outputs.guard, SideOutput)
    assert isinstance(out.outputs.bandit, SideOutput)


def test_step_advances_time_and_step():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))
    actions = _zero_actions(env)
    out = env.step(jax.random.PRNGKey(1), state, actions)
    assert float(out.state.t) == float(state.t) + cfg.dt
    assert int(out.state.step) == int(state.step) + 1


def test_step_episode_done_broadcasts_to_per_side():
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))
    actions = _zero_actions(env)
    out = env.step(jax.random.PRNGKey(1), state, actions)
    assert out.outputs.guard.done.shape == out.episode_done.shape
    assert out.outputs.bandit.done.shape == out.episode_done.shape
    assert (out.outputs.guard.done == out.episode_done).all()
    assert (out.outputs.bandit.done == out.episode_done).all()


def test_step_jit_compiles():
    """jax.jit on env.step succeeds — sanity check for pytree purity."""
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    state, _outs = env.reset(jax.random.PRNGKey(0))
    actions = _zero_actions(env)
    jit_step = jax.jit(env.step)
    out = jit_step(jax.random.PRNGKey(1), state, actions)
    assert out.episode_done.shape == ()


def test_reset_returns_byside_outputs():
    """Reset returns (env_state, BySide[SideOutput]). Both sides populated."""
    cfg = build_config()
    env = OrbitalGameEnv(cfg)
    state, outputs = env.reset(jax.random.PRNGKey(0))
    assert isinstance(outputs, BySide)
    assert isinstance(outputs.guard, SideOutput)
    assert isinstance(outputs.bandit, SideOutput)
    # Initial done should be False, reward 0.0.
    assert bool(outputs.guard.done) is False
    assert bool(outputs.bandit.done) is False
