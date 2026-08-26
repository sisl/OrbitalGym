from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from orbitalgym.groundstations import ContactSchedule
from orbitalgym.groundstations.contacts import (
    in_contact_now,
    precompute_contact_schedule,
)
from orbitalgym.groundstations.network import GroundStation
from orbitalgym.reference_orbit import ReferenceOrbitState


def _schedule(*windows, station_ix=None, pad_to=8):
    """Helper: build a padded ContactSchedule from a list of (start, end) tuples."""
    n = len(windows)
    pad = [(-1.0, -1.0)] * (pad_to - n)
    arr = jnp.asarray(list(windows) + pad, dtype=jnp.float32)
    if station_ix is None:
        station_ix = list(range(n)) + [-1] * (pad_to - n)
    return ContactSchedule(
        windows=arr,
        n_valid=jnp.asarray(n),
        station_ix=jnp.asarray(station_ix),
    )


def test_before_first_window():
    sch = _schedule((100.0, 200.0))
    assert not bool(in_contact_now(sch, jnp.asarray(50.0)))


def test_inside_window():
    sch = _schedule((100.0, 200.0))
    assert bool(in_contact_now(sch, jnp.asarray(150.0)))


def test_at_start_inclusive():
    sch = _schedule((100.0, 200.0))
    assert bool(in_contact_now(sch, jnp.asarray(100.0)))


def test_at_end_exclusive():
    sch = _schedule((100.0, 200.0))
    assert not bool(in_contact_now(sch, jnp.asarray(200.0)))


def test_between_windows():
    sch = _schedule((100.0, 200.0), (300.0, 400.0))
    assert not bool(in_contact_now(sch, jnp.asarray(250.0)))


def test_inside_second_window():
    sch = _schedule((100.0, 200.0), (300.0, 400.0))
    assert bool(in_contact_now(sch, jnp.asarray(350.0)))


def test_padded_sentinels_ignored():
    sch = _schedule((100.0, 200.0), pad_to=16)
    # All sentinels are (-1, -1); query at t=-0.5 must NOT match the sentinels.
    assert not bool(in_contact_now(sch, jnp.asarray(-0.5)))


def test_jit_traceable():
    sch = _schedule((100.0, 200.0))
    f = jax.jit(in_contact_now)
    assert bool(f(sch, jnp.asarray(150.0)))


def _ref_orbit():
    return ReferenceOrbitState(
        position_eci=jnp.array([7000e3, 0.0, 0.0]),
        velocity_eci=jnp.array([0.0, 7.5e3, 0.0]),
    )


def _stations():
    return (
        GroundStation(
            name="A",
            lat_deg=jnp.asarray(0.0),
            lon_deg=jnp.asarray(0.0),
            altitude_m=jnp.asarray(0.0),
            elevation_mask_deg=jnp.asarray(5.0),
        ),
        GroundStation(
            name="B",
            lat_deg=jnp.asarray(45.0),
            lon_deg=jnp.asarray(180.0),
            altitude_m=jnp.asarray(0.0),
            elevation_mask_deg=jnp.asarray(5.0),
        ),
    )


def test_precompute_aggregates_and_sorts_windows():
    # Mock brahe.find_location_accesses to return per-station windows.
    # Station A: one window (idx=0), Station B: two windows (idx=1).
    # Result should be sorted ascending by start time.
    def fake_accesses(*args, **kwargs):
        sta = kwargs.get("station_index", args[-1] if args else None)
        if sta == 0:
            return [(500.0, 800.0)]
        return [(100.0, 300.0), (1500.0, 1700.0)]

    with patch(
        "orbitalgym.groundstations.contacts._brahe_find_accesses",
        side_effect=fake_accesses,
    ):
        sch = precompute_contact_schedule(
            stations=_stations(),
            reference_orbit=_ref_orbit(),
            epoch_mjd_utc=60067.0,
            horizon_s=2000.0,
            pad_to=8,
        )

    n = int(sch.n_valid)
    assert n == 3
    starts = sch.windows[:n, 0]
    assert float(starts[0]) == pytest.approx(100.0)
    assert float(starts[1]) == pytest.approx(500.0)
    assert float(starts[2]) == pytest.approx(1500.0)
    # station tags follow the sorted order
    assert int(sch.station_ix[0]) == 1  # B at 100
    assert int(sch.station_ix[1]) == 0  # A at 500
    assert int(sch.station_ix[2]) == 1  # B at 1500


def test_precompute_raises_when_pad_too_small():
    def fake_accesses(*args, **kwargs):
        return [(100.0 * i, 100.0 * i + 50.0) for i in range(20)]

    with (
        patch(
            "orbitalgym.groundstations.contacts._brahe_find_accesses",
            side_effect=fake_accesses,
        ),
        pytest.raises(ValueError, match="exceeds pad_to"),
    ):
        precompute_contact_schedule(
            stations=_stations()[:1],
            reference_orbit=_ref_orbit(),
            epoch_mjd_utc=60067.0,
            horizon_s=10000.0,
            pad_to=4,
        )


def test_brahe_shim_returns_sorted_seconds():
    from orbitalgym.groundstations.contacts import _brahe_find_accesses

    # Reference: 500km circular, polar (97.8°) orbit — guaranteed coverage of Svalbard.
    R_E = 6378137.0  # noqa: N806 - Earth radius (physics convention)
    a = R_E + 500e3
    v = (3.986004418e14 / a) ** 0.5
    # Inclination 97.8° relative to equator: position at ascending node x-axis,
    # velocity in y-z plane with z-component sin(i), y-component cos(i).
    import math

    incl = math.radians(97.8)
    eci6 = [a, 0.0, 0.0, 0.0, v * math.cos(incl), v * math.sin(incl)]

    wins = _brahe_find_accesses(
        eci6,
        epoch_mjd_utc=60310.0,  # 2024-01-01-ish
        horizon_s=86400.0,  # 1 day
        station_lat_deg=78.2,  # Svalbard
        station_lon_deg=15.4,
        station_alt_m=0.0,
        elevation_mask_deg=5.0,
        station_index=0,
    )

    # We expect at least a few passes per day from a 500km/polar orbit over Svalbard.
    assert len(wins) > 0
    # Each window inside horizon and well-formed.
    for t0, t1 in wins:
        assert 0.0 <= t0 < t1 <= 86400.0
    # Sorted ascending
    starts = [t0 for t0, _ in wins]
    assert starts == sorted(starts)
