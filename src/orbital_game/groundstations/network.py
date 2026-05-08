"""GroundStation, ContactSchedule, GroundStationNetwork.

Geometry-layer pytrees for ground-station-gated comms. The runtime carries
only the precomputed `ContactSchedule` (stacked window arrays); the brahe-
backed `precompute_contact_schedule` builder and the runtime `in_contact_now`
predicate both live in `contacts.py`."""

from __future__ import annotations

from typing import Any

import flax.struct
import jax


@flax.struct.dataclass
class GroundStation:
    """A single ground station — Earth-fixed lat/lon/alt + elevation mask.

    `name` is a Python-level string (not a traced array) so it stays usable
    by visualization and logging without escaping the JAX trace.
    """

    name: str = flax.struct.field(pytree_node=False)
    lat_deg: jax.Array
    lon_deg: jax.Array
    altitude_m: jax.Array
    elevation_mask_deg: jax.Array

    def __post_init__(self) -> None:
        # Concrete validation lives outside the JAX trace — these are
        # construction-time checks, not jit-stable predicates. Skip when
        # any field is a tracer (e.g. flax rebuilds the pytree under jit).
        if isinstance(self.lat_deg, jax.core.Tracer):
            return
        lat = float(self.lat_deg)
        lon = float(self.lon_deg)
        alt = float(self.altitude_m)
        mask = float(self.elevation_mask_deg)
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"lat_deg must be in [-90, 90], got {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"lon_deg must be in [-180, 180], got {lon}")
        if alt < 0.0:
            raise ValueError(f"altitude_m must be >= 0, got {alt}")
        if not (0.0 <= mask < 90.0):
            raise ValueError(f"elevation_mask_deg must be in [0, 90), got {mask}")


@flax.struct.dataclass
class ContactSchedule:
    """Precomputed contact windows for one side over the episode horizon.

    `windows` is a fixed-shape padded array so `in_contact_now` runs in JIT
    without Python control flow. Sentinel rows have `t_end <= t_start` and
    are guarded by the per-row `idx < n_valid` mask.
    """

    windows: jax.Array  # (N_max, 2)  (t_start_s, t_end_s)
    n_valid: jax.Array  # () int — count of real windows
    station_ix: jax.Array  # (N_max,) which station owns each window (for viz)

    def __post_init__(self) -> None:
        # Construction-time shape and bounds checks. The precompute builder
        # enforces sort order; sentinel rows make an in-trace overlap check
        # awkward, so we skip that here. Shape checks are tracer-safe; the
        # bounds check on `n_valid` needs a concrete value, so it's skipped
        # under jit (where flax rebuilds the pytree with traced leaves).
        if self.windows.ndim < 2 or self.windows.shape[-1] != 2:
            raise ValueError(f"windows must have shape (..., 2); got {self.windows.shape}")
        n_max = self.windows.shape[0]
        if self.station_ix.shape[0] != n_max:
            raise ValueError(
                "station_ix and windows must have matching pad length; "
                f"got station_ix.shape[0]={self.station_ix.shape[0]} vs "
                f"windows.shape[0]={n_max}"
            )
        if isinstance(self.n_valid, jax.core.Tracer):
            return
        n_valid_int = int(self.n_valid)
        if not (0 <= n_valid_int <= n_max):
            raise ValueError(f"n_valid must be in [0, {n_max}]; got {n_valid_int}")


@flax.struct.dataclass
class GroundStationNetwork:
    """A side's network — static-tuple stations + their precomputed schedule.

    `schedule` is filled by `precompute_contact_schedule` (in `contacts.py`)
    at scenario construction. Construction without a schedule is allowed for
    tests that don't exercise contacts; production code goes through the
    builder."""

    stations: tuple[GroundStation, ...] = flax.struct.field(pytree_node=False)
    schedule: Any = None  # ContactSchedule | None — None for not-yet-precomputed

    def __post_init__(self) -> None:
        if not self.stations:
            raise ValueError("GroundStationNetwork requires at least one station")
