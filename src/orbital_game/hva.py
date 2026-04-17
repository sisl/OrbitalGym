"""High-Value Asset (HVA) state.

The HVA's ECI position and velocity at the scenario epoch. Epoch itself lives
on ScenarioConfig.epoch_mjd_utc — not here — because epoch is a scenario-level
time anchor shared by every vehicle, not a property of the asset.
"""
from __future__ import annotations

import flax.struct
import jax


@flax.struct.dataclass
class HVAState:
    """HVA 6D state in Earth-Centered Inertial frame at the scenario epoch.

    Canonical representation. Osculating elements are derived on demand via
    astrojax helpers and never stored as fields (avoids dual-source-of-truth bugs).
    """
    position_eci: jax.Array  # (3,) meters
    velocity_eci: jax.Array  # (3,) m/s
