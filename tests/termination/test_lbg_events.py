"""Tests for LbgEventTermination — within-step catch/breach with speed gates."""

from __future__ import annotations

import jax.numpy as jnp

from orbitalgym.termination.lbg_events import LbgEventTermination


class _Cfg:
    def __init__(self, max_steps: int = 100, dt: float = 10.0):
        self.max_steps = max_steps
        self.dt = dt


class _Side:
    def __init__(self, rtn):
        self.rtn = rtn


class _State:
    def __init__(self, guards, bandits, step: int = 0):
        self.guards = guards
        self.bandits = bandits
        self.step = jnp.asarray(step)


def _state(g_xyz, b_xyz, step=0):
    g = jnp.asarray(g_xyz, dtype=jnp.float32)
    b = jnp.asarray(b_xyz, dtype=jnp.float32)
    g_full = jnp.concatenate([g, jnp.zeros_like(g)], axis=-1)
    b_full = jnp.concatenate([b, jnp.zeros_like(b)], axis=-1)
    return _State(_Side(g_full), _Side(b_full), step=step)


_FAR_GUARD = [[5000.0, 0.0, 0.0]]


def test_slow_breach_between_samples_terminates():
    """A slow bandit that cuts the breach sphere between samples ends the episode."""
    term = LbgEventTermination(breach_radius_m=5.0, breach_speed_mps=0.5)
    prev = _state(_FAR_GUARD, [[-6.0, 4.9, 0.0]])
    nxt = _state(_FAR_GUARD, [[6.0, 4.9, 0.0]], step=1)
    assert bool(term(prev, nxt, _Cfg(dt=40.0), nxt.step))


def test_fast_breach_between_samples_does_not_terminate():
    """The same geometry traversed ten times faster fails the speed gate."""
    term = LbgEventTermination(breach_radius_m=5.0, breach_speed_mps=0.5)
    prev = _state(_FAR_GUARD, [[-6.0, 4.9, 0.0]])
    nxt = _state(_FAR_GUARD, [[6.0, 4.9, 0.0]], step=1)
    assert not bool(term(prev, nxt, _Cfg(dt=4.0), nxt.step))


def test_infinite_speed_gate_is_radius_only():
    """The default infinite gate fires on the radius alone."""
    term = LbgEventTermination(breach_radius_m=5.0)
    prev = _state(_FAR_GUARD, [[-6.0, 4.9, 0.0]])
    nxt = _state(_FAR_GUARD, [[6.0, 4.9, 0.0]], step=1)
    assert bool(term(prev, nxt, _Cfg(dt=4.0), nxt.step))


def test_catch_between_samples_terminates():
    """A guard-bandit closest approach inside the catch radius ends the episode."""
    term = LbgEventTermination(breach_radius_m=5.0, catch_radius_m=50.0)
    prev = _state([[0.0, 0.0, 0.0]], [[-400.0, 30.0, 0.0]])
    nxt = _state([[0.0, 0.0, 0.0]], [[400.0, 30.0, 0.0]], step=1)
    assert bool(term(prev, nxt, _Cfg(), nxt.step))


def test_catch_disabled_by_zero_radius():
    term = LbgEventTermination(breach_radius_m=5.0, catch_radius_m=0.0)
    prev = _state([[0.0, 0.0, 0.0]], [[-400.0, 30.0, 0.0]])
    nxt = _state([[0.0, 0.0, 0.0]], [[400.0, 30.0, 0.0]], step=1)
    assert not bool(term(prev, nxt, _Cfg(), nxt.step))


def test_max_steps_still_terminates():
    term = LbgEventTermination(breach_radius_m=5.0, catch_radius_m=50.0)
    prev = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]], step=9)
    nxt = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]], step=10)
    assert bool(term(prev, nxt, _Cfg(max_steps=10), nxt.step))


def test_quiet_step_does_not_terminate():
    term = LbgEventTermination(breach_radius_m=5.0, catch_radius_m=50.0)
    prev = _state(_FAR_GUARD, [[1000.0, 0.0, 0.0]])
    nxt = _state(_FAR_GUARD, [[1000.0, 5.0, 0.0]], step=1)
    assert not bool(term(prev, nxt, _Cfg(), nxt.step))
