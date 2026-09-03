"""Contact windows must agree with a direct elevation computation."""

import astrojax
import jax.numpy as jnp
import numpy as np
import pytest

from orbitalgym.groundstations import GroundStation
from orbitalgym.groundstations.contacts import in_contact_now, precompute_contact_schedule
from orbitalgym.reference_orbit import ReferenceOrbitState

MU = 3.986004418e14
R_EARTH = 6378137.0
OMEGA_EARTH = 7.2921159e-5  # rad/s

_MJD_EPOCH = astrojax.Epoch(1858, 11, 17, 0, 0, 0.0)


def _epoch_from_mjd(mjd: float) -> astrojax.Epoch:
    return _MJD_EPOCH + float(mjd) * 86400.0


def _station(lat_deg: float, lon_deg: float, mask_deg: float) -> GroundStation:
    return GroundStation(
        name="S",
        lat_deg=jnp.asarray(lat_deg),
        lon_deg=jnp.asarray(lon_deg),
        altitude_m=jnp.asarray(0.0),
        elevation_mask_deg=jnp.asarray(mask_deg),
    )


def _elevation_deg(r_eci: np.ndarray, t_s: float, epoch_mjd: float, lat_deg, lon_deg) -> float:
    """Elevation of an ECI position from a ground point, using GMST-only rotation."""
    gmst0 = float(_epoch_from_mjd(epoch_mjd).gmst())  # radians at epoch
    theta = gmst0 + OMEGA_EARTH * t_s
    c, s = np.cos(theta), np.sin(theta)
    r_ecef = np.array([c * r_eci[0] + s * r_eci[1], -s * r_eci[0] + c * r_eci[1], r_eci[2]])
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    site = R_EARTH * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    up = site / np.linalg.norm(site)
    los = r_ecef - site
    return float(np.rad2deg(np.arcsin(np.dot(los, up) / np.linalg.norm(los))))


@pytest.mark.parametrize("lat_deg,lon_deg", [(0.0, 0.0), (45.0, 10.0), (-35.0, 149.0)])
def test_schedule_matches_direct_elevation(lat_deg, lon_deg):
    epoch = 60067.0
    horizon = 4.0 * 5828.5
    ref = ReferenceOrbitState(
        position_eci=jnp.array([7000e3, 0.0, 0.0]),
        velocity_eci=jnp.array(
            [0.0, 7.5e3 * np.cos(np.deg2rad(45.0)), 7.5e3 * np.sin(np.deg2rad(45.0))]
        ),
    )
    schedule = precompute_contact_schedule((_station(lat_deg, lon_deg, 5.0),), ref, epoch, horizon)

    # Independent propagation: two-body Keplerian via astrojax at 30 s steps.
    eop = astrojax.zero_eop()
    dyn = astrojax.create_orbit_dynamics(eop, _epoch_from_mjd(epoch))
    x = np.asarray(jnp.concatenate([ref.position_eci, ref.velocity_eci]), dtype=float)
    ts = np.arange(0.0, horizon, 30.0)
    mismatches = 0
    checked = 0
    for t in ts:
        el = _elevation_deg(x[:3], t, epoch, lat_deg, lon_deg)
        predicted = bool(in_contact_now(schedule, jnp.asarray(t, dtype=jnp.float32)))
        direct = el > 5.0
        # Ignore ticks within 60 s of a direct-computed edge; the search step is 60 s.
        if abs(el - 5.0) > 0.5:
            checked += 1
            mismatches += int(predicted != direct)
        x = np.asarray(astrojax.rk4_step(dyn, float(t), jnp.asarray(x), 30.0).state)
    assert checked > 100
    assert mismatches == 0, f"{mismatches} of {checked} ticks disagree with direct elevation"
