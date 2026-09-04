"""Tests for LbgEventTermination — within-step catch/breach with speed gates."""

from __future__ import annotations

import jax.numpy as jnp

from orbitalgym.reference_orbit import ReferenceOrbitState, mean_motion
from orbitalgym.termination.lbg_events import LbgEventTermination

_REFERENCE_ORBIT = ReferenceOrbitState(
    position_eci=jnp.array([7000e3, 0.0, 0.0]),
    velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
)
_MEAN_MOTION = float(mean_motion(_REFERENCE_ORBIT))


class _Cfg:
    def __init__(self, max_steps: int = 100, dt: float = 10.0):
        self.max_steps = max_steps
        self.dt = dt
        self.reference_orbit = _REFERENCE_ORBIT


class _Side:
    def __init__(self, rtn, propellant_mass=None):
        self.rtn = rtn
        self.propellant_mass = propellant_mass


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


def _rtn_state(guard_rtn, bandit_rtn, propellant, step=0):
    guards = _Side(jnp.asarray(guard_rtn, dtype=jnp.float32))
    bandits = _Side(
        jnp.asarray(bandit_rtn, dtype=jnp.float32),
        jnp.asarray(propellant, dtype=jnp.float32),
    )
    return _State(guards, bandits, step=step)


_FAR_GUARD_RTN = [[5000.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


def test_bandit_beyond_escape_radius_is_repelled():
    """A bandit 6 km out with a 5 km escape radius ends the episode."""
    term = LbgEventTermination(breach_radius_m=5.0, escape_radius_m=5000.0)
    prev = _state(_FAR_GUARD, [[6000.0, 0.0, 0.0]])
    nxt = _state(_FAR_GUARD, [[6000.0, 10.0, 0.0]], step=1)
    assert bool(term(prev, nxt, _Cfg(), nxt.step))


def test_bandit_inside_escape_radius_is_not_repelled():
    term = LbgEventTermination(breach_radius_m=5.0, escape_radius_m=5000.0)
    prev = _state(_FAR_GUARD, [[4000.0, 0.0, 0.0]])
    nxt = _state(_FAR_GUARD, [[4000.0, 10.0, 0.0]], step=1)
    assert not bool(term(prev, nxt, _Cfg(), nxt.step))


def test_empty_tank_on_bounded_ellipse_is_repelled():
    """An out-of-propellant bandit on a 3 km bounded ellipse can never reach the lady."""
    term = LbgEventTermination(breach_radius_m=15.0, repel_on_empty_tank=True)
    ellipse = [[3000.0, 0.0, 0.0, 0.0, -2.0 * _MEAN_MOTION * 3000.0, 0.0]]
    prev = _rtn_state(_FAR_GUARD_RTN, ellipse, [0.0])
    nxt = _rtn_state(_FAR_GUARD_RTN, ellipse, [0.0], step=1)
    assert bool(term(prev, nxt, _Cfg(max_steps=200), nxt.step))


def test_empty_tank_still_coasting_toward_the_lady_is_not_repelled():
    """A 0.1 m/s inbound drift from 30 m still reaches the breach sphere."""
    term = LbgEventTermination(breach_radius_m=15.0, repel_on_empty_tank=True)
    inbound = [[30.0, 0.0, 0.0, -0.1, 0.0, 0.0]]
    prev = _rtn_state(_FAR_GUARD_RTN, inbound, [0.0])
    nxt = _rtn_state(_FAR_GUARD_RTN, inbound, [0.0], step=1)
    assert not bool(term(prev, nxt, _Cfg(max_steps=200), nxt.step))


def test_fuelled_bandit_on_bounded_ellipse_is_not_repelled():
    """Propellant left means the bandit can still manoeuvre, so it is not repelled."""
    term = LbgEventTermination(breach_radius_m=15.0, repel_on_empty_tank=True)
    ellipse = [[3000.0, 0.0, 0.0, 0.0, -2.0 * _MEAN_MOTION * 3000.0, 0.0]]
    prev = _rtn_state(_FAR_GUARD_RTN, ellipse, [1.0])
    nxt = _rtn_state(_FAR_GUARD_RTN, ellipse, [1.0], step=1)
    assert not bool(term(prev, nxt, _Cfg(max_steps=200), nxt.step))


def test_repelled_needs_every_bandit_repelled():
    """One bandit still inside the escape radius keeps the episode running."""
    term = LbgEventTermination(breach_radius_m=5.0, escape_radius_m=5000.0)
    bandits = [[6000.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]
    prev = _state(_FAR_GUARD, bandits)
    nxt = _state(_FAR_GUARD, bandits, step=1)
    assert not bool(term(prev, nxt, _Cfg(), nxt.step))


def test_repel_disabled_by_default():
    term = LbgEventTermination(breach_radius_m=5.0)
    prev = _state(_FAR_GUARD, [[60000.0, 0.0, 0.0]])
    nxt = _state(_FAR_GUARD, [[60000.0, 10.0, 0.0]], step=1)
    assert not bool(term(prev, nxt, _Cfg(), nxt.step))
