"""Reference orbit state — the dynamical reference frame for HCW linearization.

The reference orbit's ECI position and velocity at the scenario epoch. Epoch itself
lives on ScenarioConfig.epoch_mjd_utc — not here — because epoch is a scenario-level
time anchor shared by every vehicle, not a property of the dynamical anchor.

Construct either directly from Cartesian ECI vectors, or from classical Keplerian
elements via :meth:`ReferenceOrbitState.from_keplerian` (which delegates to
``astrojax.state_koe_to_eci`` and accepts angles in degrees by default).
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp
from astrojax import state_koe_to_eci

MU_EARTH = 3.986004418e14


@flax.struct.dataclass
class ReferenceOrbitState:
    """Reference orbit 6D state in Earth-Centered Inertial frame at the scenario epoch.

    The dynamical anchor that HCW linearizes about. Canonical representation;
    osculating elements derived on demand via astrojax helpers.
    """

    position_eci: jax.Array  # (3,) meters
    velocity_eci: jax.Array  # (3,) m/s

    @classmethod
    def from_keplerian(
        cls,
        semi_major_axis_m: float,
        eccentricity: float,
        inclination: float,
        raan: float,
        argument_of_perigee: float,
        mean_anomaly: float,
        as_degrees: bool = True,
    ) -> ReferenceOrbitState:
        """Build a ReferenceOrbitState from classical Keplerian elements.

        Element convention matches astrojax / `sampling.side.RelativeKeplerian`:
        ``[a, e, i, RAAN, omega, M]`` with ``M`` the mean anomaly. Semi-major
        axis is in meters; angle units are controlled by ``as_degrees``.
        Conversion delegates to ``astrojax.state_koe_to_eci`` (Montenbruck &
        Gill *Satellite Orbits*, Eq. 2.43–2.44).

        Args:
            semi_major_axis_m: Semi-major axis ``a`` in meters. Must be > 0.
            eccentricity: Eccentricity ``e``. Must satisfy ``0 <= e < 1``
                (elliptical / circular only; parabolic and hyperbolic rejected).
            inclination: Inclination ``i``.
            raan: Right ascension of ascending node ``Ω``.
            argument_of_perigee: Argument of perigee ``ω``.
            mean_anomaly: Mean anomaly ``M``.
            as_degrees: If ``True`` (default), interpret all angle inputs as
                degrees. If ``False``, interpret them as radians.

        Returns:
            ReferenceOrbitState with ``position_eci`` (3,) meters and
            ``velocity_eci`` (3,) m/s.

        Raises:
            ValueError: If ``semi_major_axis_m <= 0`` or eccentricity is
                outside ``[0, 1)``.

        Examples:
            ```python
            from orbital_game.reference_orbit import ReferenceOrbitState

            # 500 km circular equatorial orbit, periapsis on the +x axis.
            ref = ReferenceOrbitState.from_keplerian(
                semi_major_axis_m=6878e3,
                eccentricity=0.0,
                inclination=0.0,
                raan=0.0,
                argument_of_perigee=0.0,
                mean_anomaly=0.0,
            )
            ```
        """
        if semi_major_axis_m <= 0.0:
            raise ValueError(
                f"semi_major_axis_m must be > 0 m; got {semi_major_axis_m!r}. "
                "Closed orbits require a strictly positive semi-major axis."
            )
        if not (0.0 <= eccentricity < 1.0):
            raise ValueError(
                f"eccentricity must be in [0, 1) for elliptical/circular orbits; "
                f"got {eccentricity!r}. Parabolic (e=1) and hyperbolic (e>1) "
                "trajectories are not supported as reference orbits."
            )
        koe = jnp.array(
            [
                semi_major_axis_m,
                eccentricity,
                inclination,
                raan,
                argument_of_perigee,
                mean_anomaly,
            ]
        )
        eci = state_koe_to_eci(koe, use_degrees=as_degrees)
        return cls(position_eci=eci[:3], velocity_eci=eci[3:])


def mean_motion(ref: ReferenceOrbitState) -> jax.Array:
    """Orbital mean motion of the reference orbit via vis-viva: 1/a = 2/r - v²/μ.

    Returns sqrt(μ/a³). Same value used by both the env's HCW dynamics and
    `RelativeEllipse`'s sampler — keeping these in sync is required so sampled
    ICs satisfy the dynamics' bounded-relative-orbit condition.
    """
    r = jnp.linalg.norm(ref.position_eci)
    v2 = jnp.sum(ref.velocity_eci**2)
    a = 1.0 / (2.0 / r - v2 / MU_EARTH)
    return jnp.sqrt(MU_EARTH / a**3)
