"""Contact-schedule precompute (brahe) and runtime predicate (JAX).

The runtime predicate `in_contact_now` is pure JAX and has no brahe
dependency. The `precompute_contact_schedule` builder is called once at
scenario construction and uses brahe (a core dependency of OrbitalGym)
to find satellite/ground-station accesses over the episode horizon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import brahe as _brahe
import jax
import jax.numpy as jnp

from orbitalgym.groundstations.network import ContactSchedule

if TYPE_CHECKING:
    from orbitalgym.groundstations.network import GroundStation
    from orbitalgym.reference_orbit import ReferenceOrbitState


def in_contact_now(schedule: ContactSchedule, t: jax.Array) -> jax.Array:
    """Scalar bool: is `t` inside any valid contact window?

    Open-closed semantics: `t_start <= t < t_end`. Pure JAX — single
    `jnp.any` over the padded `(N_max, 2)` array, gated by an
    `idx < n_valid` mask so sentinel rows never match.
    """
    starts = schedule.windows[:, 0]
    ends = schedule.windows[:, 1]
    idx = jnp.arange(starts.shape[0])
    valid = idx < schedule.n_valid
    inside = (starts <= t) & (t < ends)
    return jnp.any(valid & inside)


def _brahe_find_accesses(
    reference_orbit_eci6: list[float],
    epoch_mjd_utc: float,
    horizon_s: float,
    station_lat_deg: float,
    station_lon_deg: float,
    station_alt_m: float,
    elevation_mask_deg: float,
    station_index: int,
) -> list[tuple[float, float]]:
    """Compute contact windows for one station against the reference orbit.

    Returns a list of (t_start_s, t_end_s) tuples, where times are seconds
    relative to the episode start (`epoch_mjd_utc`).

    Wrapped so tests can monkey-patch this single function rather than
    mocking the brahe import.
    """
    del station_index  # unused inside the shim; preserved for test ergonomics

    # brahe needs EOP initialized before it can rotate ECI<->ECEF for access
    # checks. Use the zero static provider — fine for short-horizon scenarios
    # where polar-motion / UT1-UTC corrections are well below the contact
    # boundary precision we care about. Production use can override globally.
    _ensure_eop_initialized()

    # MJD-UTC -> brahe.Epoch
    import numpy as np

    # pyrefly: ignore[bad-argument-type]
    epc_start = _brahe.Epoch.from_mjd(float(epoch_mjd_utc), _brahe.TimeSystem.UTC)
    # pyrefly: ignore[unsupported-operation]
    epc_end = epc_start + float(horizon_s)

    # Propagator from ECI 6-vector. step_size=60s is a reasonable default;
    # access-window search adapts internally.
    state = np.asarray(reference_orbit_eci6, dtype=np.float64)
    prop = _brahe.KeplerianPropagator.from_eci(epc_start, state, 60.0)

    # PointLocation expects (lon, lat, alt) in degrees / meters.
    location = _brahe.PointLocation(
        float(station_lon_deg), float(station_lat_deg), float(station_alt_m)
    )
    constraint = _brahe.ElevationConstraint(float(elevation_mask_deg))

    # pyrefly: ignore[bad-argument-type]
    windows = _brahe.location_accesses(location, prop, epc_start, epc_end, constraint)

    # AccessWindow.t_start / t_end are Epoch objects. Epoch - Epoch -> float seconds.
    # pyrefly: ignore[unsupported-operation]
    return [(float(w.t_start - epc_start), float(w.t_end - epc_start)) for w in windows]


_EOP_INITIALIZED = False


def _ensure_eop_initialized() -> None:
    """One-time EOP setup. brahe panics if accessed before initialization."""
    global _EOP_INITIALIZED
    if _EOP_INITIALIZED:
        return
    _brahe.set_global_eop_provider_from_static_provider(_brahe.StaticEOPProvider.from_zero())
    _EOP_INITIALIZED = True


def precompute_contact_schedule(
    stations: tuple[GroundStation, ...],
    reference_orbit: ReferenceOrbitState,
    epoch_mjd_utc: float,
    horizon_s: float,
    *,
    pad_to: int = 256,
) -> ContactSchedule:
    """Build a ContactSchedule from per-station brahe queries.

    Calls brahe once per station, concatenates the per-station window
    lists, sorts ascending by start time, pads to pad_to rows with
    sentinels.

    Raises ValueError if the total window count exceeds pad_to.
    """
    ref6 = jnp.concatenate([reference_orbit.position_eci, reference_orbit.velocity_eci]).tolist()

    pairs: list[tuple[float, float, int]] = []
    for ix, st in enumerate(stations):
        wins = _brahe_find_accesses(
            ref6,
            epoch_mjd_utc,
            horizon_s,
            float(st.lat_deg),
            float(st.lon_deg),
            float(st.altitude_m),
            float(st.elevation_mask_deg),
            ix,
        )
        for t0, t1 in wins:
            pairs.append((t0, t1, ix))

    pairs.sort(key=lambda p: p[0])
    n = len(pairs)
    if n > pad_to:
        raise ValueError(
            f"Total contact windows ({n}) exceeds pad_to={pad_to}. "
            f"Increase pad_to or shorten horizon_s."
        )

    windows = jnp.full((pad_to, 2), -1.0, dtype=jnp.float32)
    station_ix = jnp.full((pad_to,), -1, dtype=jnp.int32)
    if n > 0:
        rows = jnp.asarray([(t0, t1) for t0, t1, _ in pairs], dtype=jnp.float32)
        ixs = jnp.asarray([s for _, _, s in pairs], dtype=jnp.int32)
        windows = windows.at[:n].set(rows)
        station_ix = station_ix.at[:n].set(ixs)

    return ContactSchedule(
        windows=windows,
        n_valid=jnp.asarray(n),
        station_ix=station_ix,
    )
