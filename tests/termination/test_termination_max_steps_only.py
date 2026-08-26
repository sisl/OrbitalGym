"""Tests for MaxStepsOnly — pure step-cap termination."""

from __future__ import annotations

import jax.numpy as jnp

from orbitalgym.termination.reference import MaxStepsOnly


class _Cfg:
    """Minimal stand-in for ScenarioConfig (only `max_steps` is read)."""

    def __init__(self, max_steps: int):
        self.max_steps = max_steps


class _State:
    def __init__(self, step: int):
        self.step = jnp.asarray(step)


def test_terminates_at_max_steps():
    term = MaxStepsOnly()
    done = term(state=_State(10), params=_Cfg(max_steps=10), t=jnp.asarray(0.0))
    assert bool(done)


def test_terminates_past_max_steps():
    term = MaxStepsOnly()
    done = term(state=_State(11), params=_Cfg(max_steps=10), t=jnp.asarray(0.0))
    assert bool(done)


def test_does_not_terminate_below_max_steps():
    term = MaxStepsOnly()
    done = term(state=_State(5), params=_Cfg(max_steps=10), t=jnp.asarray(0.0))
    assert not bool(done)
