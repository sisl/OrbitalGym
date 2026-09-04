"""Tests for within-step closest-approach proximity events.

Covers the segment model used by LBG catch/breach events: a straight-line
coast from the pre-step relative position with the chord velocity, gated on
both a miss distance and a relative speed.
"""

from __future__ import annotations

import jax.numpy as jnp

from orbitalgym.dynamics.hcw import hcw_rtn_stm
from orbitalgym.games.proximity import (
    ballistic_breach_possible,
    closest_approach,
    proximity_event,
)
from orbitalgym.reference_orbit import ReferenceOrbitState, mean_motion


def _segment(r_prev, r_next, dt):
    r0 = jnp.asarray(r_prev, dtype=jnp.float32)
    r1 = jnp.asarray(r_next, dtype=jnp.float32)
    return r0, (r1 - r0) / dt


def test_fast_flyby_through_sphere_is_not_event():
    """Passing inside the radius at high relative speed is not an event."""
    r0, v = _segment((-20.0, 3.0, 0.0), (20.0, 3.0, 0.0), 10.0)
    d, speed = closest_approach(r0, v, 10.0)
    assert float(d) == 3.0
    assert float(speed) == 4.0
    assert not bool(proximity_event(r0, v, 10.0, radius_m=5.0, speed_mps=0.5))


def test_slow_pass_between_samples_is_event():
    """A slow pass whose closest approach falls between samples is an event."""
    r0, v = _segment((-4.0, 3.0, 0.0), (4.0, 3.0, 0.0), 20.0)
    d, speed = closest_approach(r0, v, 20.0)
    assert float(d) == 3.0
    assert abs(float(speed) - 0.4) < 1e-6
    assert bool(proximity_event(r0, v, 20.0, radius_m=5.0, speed_mps=0.5))


def test_endpoints_outside_segment_inside():
    """Both endpoints sit outside the sphere while the segment cuts through it."""
    r0, v = _segment((-6.0, 4.9, 0.0), (6.0, 4.9, 0.0), 40.0)
    d, speed = closest_approach(r0, v, 40.0)
    assert float(jnp.linalg.norm(r0)) > 7.7
    assert float(jnp.linalg.norm(r0 + v * 40.0)) > 7.7
    assert abs(float(d) - 4.9) < 1e-5
    assert abs(float(speed) - 0.3) < 1e-6
    assert bool(proximity_event(r0, v, 40.0, radius_m=5.0, speed_mps=0.5))


def test_zero_velocity_uses_endpoint_distance():
    """A stationary relative position yields its own norm and no NaN."""
    r0 = jnp.asarray([3.0, 4.0, 0.0])
    v = jnp.zeros(3)
    d, speed = closest_approach(r0, v, 10.0)
    assert float(d) == 5.0
    assert float(speed) == 0.0
    assert not jnp.isnan(d)
    assert bool(proximity_event(r0, v, 10.0, radius_m=6.0, speed_mps=0.5))


def test_batched_inputs_broadcast():
    """Leading batch axes are preserved."""
    r0 = jnp.asarray([[-20.0, 3.0, 0.0], [-4.0, 3.0, 0.0]])
    v = jnp.asarray([[4.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    d, speed = closest_approach(r0, v, 10.0)
    assert d.shape == (2,)
    assert speed.shape == (2,)
    events = proximity_event(r0, v, 10.0, radius_m=5.0, speed_mps=0.5)
    assert events.shape == (2,)
    assert not bool(events[0])
    assert bool(events[1])


def _phi(dt: float = 10.0):
    ref = ReferenceOrbitState(
        position_eci=jnp.array([7000e3, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
    )
    return hcw_rtn_stm(float(mean_motion(ref)), dt)


def test_bounded_ellipse_never_reaches_the_origin():
    """A 3 km bounded relative ellipse cannot coast into the breach sphere."""
    ref = ReferenceOrbitState(
        position_eci=jnp.array([7000e3, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
    )
    n = float(mean_motion(ref))
    x = jnp.array([3000.0, 0.0, 0.0, 0.0, -2.0 * n * 3000.0, 0.0])
    possible = ballistic_breach_possible(
        x, _phi(), jnp.asarray(200), 200, 10.0, radius_m=15.0, speed_mps=jnp.inf
    )
    assert not bool(possible)


def test_inbound_drift_reaches_the_breach_sphere():
    """A 0.1 m/s radial drift from 30 m coasts inside a 15 m breach sphere."""
    x = jnp.array([30.0, 0.0, 0.0, -0.1, 0.0, 0.0])
    possible = ballistic_breach_possible(
        x, _phi(), jnp.asarray(200), 200, 10.0, radius_m=15.0, speed_mps=jnp.inf
    )
    assert bool(possible)


def test_remaining_steps_mask_excludes_late_approach():
    """The same drift is out of reach when only ten steps remain."""
    x = jnp.array([30.0, 0.0, 0.0, -0.1, 0.0, 0.0])
    possible = ballistic_breach_possible(
        x, _phi(), jnp.asarray(10), 200, 10.0, radius_m=15.0, speed_mps=jnp.inf
    )
    assert not bool(possible)


def test_ballistic_scan_maps_over_leading_axes():
    """A stacked bandit state returns one flag per vehicle."""
    x = jnp.stack(
        [
            jnp.array([30.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
            jnp.array([3000.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )
    possible = ballistic_breach_possible(
        x, _phi(), jnp.asarray(200), 200, 10.0, radius_m=15.0, speed_mps=jnp.inf
    )
    assert possible.shape == (2,)
    assert bool(possible[0])
    assert not bool(possible[1])
