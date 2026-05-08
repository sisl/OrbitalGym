"""Tests for MaxDistanceTermination + AnyOfTermination.

Covers:
  - Below threshold → no termination.
  - Single vehicle above threshold → terminates.
  - Bandit-only drift triggers (guards inside the cap).
  - Composition via AnyOfTermination short-circuits OR over wrapped terms.
"""

from __future__ import annotations

import jax.numpy as jnp

from orbital_game.termination.max_distance import (
    AnyOfTermination,
    MaxDistanceTermination,
)
from orbital_game.termination.reference import MaxStepsOnly


class _Cfg:
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps


class _Side:
    """Stand-in for guards/bandits side-state with an `rtn` field."""

    def __init__(self, rtn: jnp.ndarray):
        self.rtn = rtn


class _RtSide:
    """Stand-in for an `rt` (2D) side-state."""

    def __init__(self, rt: jnp.ndarray):
        self.rt = rt


class _State:
    def __init__(self, guards, bandits, step: int = 0):
        self.guards = guards
        self.bandits = bandits
        self.step = jnp.asarray(step)


def _state(g_xyz, b_xyz, step=0):
    g = jnp.asarray(g_xyz, dtype=jnp.float32)
    b = jnp.asarray(b_xyz, dtype=jnp.float32)
    # Pad to (N, 6) — termination only reads positions [:, :3].
    g_full = jnp.concatenate([g, jnp.zeros_like(g)], axis=-1)
    b_full = jnp.concatenate([b, jnp.zeros_like(b)], axis=-1)
    return _State(_Side(g_full), _Side(b_full), step=step)


def test_max_distance_below_threshold():
    term = MaxDistanceTermination(max_distance_m=10000.0)
    s = _state([[100.0, 200.0, 0.0]], [[3000.0, 0.0, 0.0]])
    done = term(state=s, params=_Cfg(), t=jnp.asarray(0.0))
    assert not bool(done)


def test_max_distance_bandit_drifted_terminates():
    term = MaxDistanceTermination(max_distance_m=10000.0)
    # Bandit at 30km — should trigger.
    s = _state([[100.0, 200.0, 0.0]], [[30000.0, 0.0, 0.0]])
    done = term(state=s, params=_Cfg(), t=jnp.asarray(0.0))
    assert bool(done)


def test_max_distance_guard_drifted_terminates():
    term = MaxDistanceTermination(max_distance_m=10000.0)
    # Guard at 12km, bandit nearby.
    s = _state([[12000.0, 0.0, 0.0]], [[1000.0, 0.0, 0.0]])
    done = term(state=s, params=_Cfg(), t=jnp.asarray(0.0))
    assert bool(done)


def test_max_distance_any_of_composition_with_max_steps():
    """AnyOfTermination fires when any branch fires (bandit drift OR step cap)."""
    drift = MaxDistanceTermination(max_distance_m=10000.0)
    steps = MaxStepsOnly()
    term = AnyOfTermination((drift, steps))

    # Both quiet — no termination.
    s_quiet = _state([[0.0, 0.0, 0.0]], [[1000.0, 0.0, 0.0]], step=3)
    assert not bool(term(state=s_quiet, params=_Cfg(max_steps=10), t=jnp.asarray(0.0)))

    # Step cap fires.
    s_steps = _state([[0.0, 0.0, 0.0]], [[1000.0, 0.0, 0.0]], step=10)
    assert bool(term(state=s_steps, params=_Cfg(max_steps=10), t=jnp.asarray(0.0)))

    # Drift fires (steps still under).
    s_drift = _state([[0.0, 0.0, 0.0]], [[20000.0, 0.0, 0.0]], step=3)
    assert bool(term(state=s_drift, params=_Cfg(max_steps=10), t=jnp.asarray(0.0)))


def test_max_distance_rt_side_state_supported():
    """Sides that carry only `rt` (2D) instead of `rtn` are handled too."""
    term = MaxDistanceTermination(max_distance_m=5000.0)
    g = jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    b = jnp.asarray([[6000.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    state = _State(_RtSide(g), _RtSide(b))
    done = term(state=state, params=_Cfg(), t=jnp.asarray(0.0))
    assert bool(done)
