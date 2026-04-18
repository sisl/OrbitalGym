"""Tests for termination/reference.py — MaxStepsOrBreach."""

import jax.numpy as jnp

from orbital_game.termination.reference import MaxStepsOrBreach


def test_max_steps_triggers_at_max_step_count():
    term = MaxStepsOrBreach(max_steps=10, breach_distance_m=0.0)

    class _St:
        step = jnp.array(10)
        defenders = type("D", (), {"rtn": jnp.zeros((1, 6))})()

    done = term(state=_St(), params=None, t=jnp.array(0.0))
    assert bool(done)


def test_breach_triggers_when_defender_closer_than_threshold():
    term = MaxStepsOrBreach(max_steps=1000, breach_distance_m=50.0)

    class _St:
        step = jnp.array(1)
        defenders = type("D", (), {"rtn": jnp.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])})()

    done = term(state=_St(), params=None, t=jnp.array(0.0))
    assert bool(done)


def test_no_termination_when_far_and_under_max_steps():
    term = MaxStepsOrBreach(max_steps=1000, breach_distance_m=50.0)

    class _St:
        step = jnp.array(5)
        defenders = type("D", (), {"rtn": jnp.array([[1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])})()

    done = term(state=_St(), params=None, t=jnp.array(0.0))
    assert not bool(done)
