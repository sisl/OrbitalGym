# Core API

Top-level environment, configuration, and rollout primitives.

## `OrbitalGymEnv`

::: orbitalgym.env.core.OrbitalGymEnv

## `EnvState`

::: orbitalgym.env.core.EnvState

## `SingleAgentView`

::: orbitalgym.env.single_agent.SingleAgentView

## `ScenarioConfig`

::: orbitalgym.config.ScenarioConfig

## `VehicleParamsSpec`

::: orbitalgym.config.VehicleParamsSpec

## `ReferenceOrbitState`

::: orbitalgym.reference_orbit.ReferenceOrbitState

The canonical representation is Cartesian ECI (`position_eci`, `velocity_eci`).
You can construct it either directly with those arrays, or from classical
Keplerian elements via `ReferenceOrbitState.from_keplerian(...)` — angles
default to degrees; pass `as_degrees=False` for radians. The Keplerian
constructor delegates to `astrojax.state_koe_to_eci` and validates that
`semi_major_axis_m > 0` and `0 ≤ eccentricity < 1`.

```python
from orbitalgym.reference_orbit import ReferenceOrbitState

# Cartesian (canonical):
ref_cart = ReferenceOrbitState(
    position_eci=jnp.array([7000e3, 0.0, 0.0]),
    velocity_eci=jnp.array([0.0, 7546.05, 0.0]),
)

# Keplerian (degrees by default):
ref_kep = ReferenceOrbitState.from_keplerian(
    semi_major_axis_m=7000e3,
    eccentricity=0.001,
    inclination=51.6,
    raan=0.0,
    argument_of_perigee=0.0,
    mean_anomaly=0.0,
)
```

## Rollout

Two drivers: `rollout` is obs-only (`agent_view = obs`); `belief_rollout`
threads per-side beliefs (`agent_view = belief`). Both compile to a
single `lax.scan`-based JIT region.

::: orbitalgym.rollout.rollout

::: orbitalgym.rollout.rollout_single_agent

::: orbitalgym.rollout.belief_rollout

::: orbitalgym.rollout.episode_mask

## Precision

::: orbitalgym.set_precision

## Logging

::: orbitalgym.logging.writer.save_run

::: orbitalgym.logging.reader.load_run
