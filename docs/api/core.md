# Core API

Top-level environment, configuration, and rollout primitives.

## `OrbitalGameEnv`

::: orbital_game.env.core.OrbitalGameEnv

## `EnvState`

::: orbital_game.env.core.EnvState

## `SingleAgentView`

::: orbital_game.env.single_agent.SingleAgentView

## `ScenarioConfig`

::: orbital_game.config.ScenarioConfig

## `VehicleParamsSpec`

::: orbital_game.config.VehicleParamsSpec

## `ReferenceOrbitState`

::: orbital_game.reference_orbit.ReferenceOrbitState

The canonical representation is Cartesian ECI (`position_eci`, `velocity_eci`).
You can construct it either directly with those arrays, or from classical
Keplerian elements via `ReferenceOrbitState.from_keplerian(...)` — angles
default to degrees; pass `as_degrees=False` for radians. The Keplerian
constructor delegates to `astrojax.state_koe_to_eci` and validates that
`semi_major_axis_m > 0` and `0 ≤ eccentricity < 1`.

```python
from orbital_game.reference_orbit import ReferenceOrbitState

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

::: orbital_game.rollout.rollout

::: orbital_game.rollout.rollout_single_agent

::: orbital_game.rollout.belief_rollout

::: orbital_game.rollout.episode_mask

## Precision

::: orbital_game.set_precision

## Logging

::: orbital_game.logging.writer.save_run

::: orbital_game.logging.reader.load_run
