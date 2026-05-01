"""High-Value Asset (HVA) state.

The HVA's ECI position and velocity at the scenario epoch. Epoch itself lives
on ScenarioConfig.epoch_mjd_utc — not here — because epoch is a scenario-level
time anchor shared by every vehicle, not a property of the asset.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

MU_EARTH = 3.986004418e14


@flax.struct.dataclass
class HVAState:
    """HVA 6D state in Earth-Centered Inertial frame at the scenario epoch.

    Canonical representation. Osculating elements are derived on demand via
    astrojax helpers and never stored as fields (avoids dual-source-of-truth bugs).
    """

    position_eci: jax.Array  # (3,) meters
    velocity_eci: jax.Array  # (3,) m/s


def mean_motion(hva: HVAState) -> jax.Array:
    """Orbital mean motion of the HVA via vis-viva: 1/a = 2/r - v²/μ.

    Returns sqrt(μ/a³). Same value used by both the env's HCW dynamics and
    `RelativeEllipse`'s sampler — keeping these in sync is required so sampled
    ICs satisfy the dynamics' bounded-relative-orbit condition.
    """
    r = jnp.linalg.norm(hva.position_eci)
    v2 = jnp.sum(hva.velocity_eci**2)
    a = 1.0 / (2.0 / r - v2 / MU_EARTH)
    return jnp.sqrt(MU_EARTH / a**3)
