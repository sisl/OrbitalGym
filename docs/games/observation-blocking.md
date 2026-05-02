# Observation-Blocking

The Bandit blocks the Guard's view of an Earth surface target — but only when the target is observable (Guard above `min_elevation_deg` from the target).

## What this game models

A spaceborne ISR (intelligence, surveillance, reconnaissance) denial scenario. The Guard satellite is tasked with observing a specific Earth surface target (e.g. a lat/lon location). The Bandit's job is to position itself to occlude the Guard's line-of-sight to the target — but only when the target is actually visible from the Guard. If the target is below the horizon (low elevation), the Bandit gets no reward regardless of geometry.

## Reward formula

`scope = PER_SIDE`, zero-sum:

```text
elevation = elevation of (guard → target)              # degrees
visible = elevation >= min_elevation_deg

angle = ∠(target→guard, target→bandit)
in_front = |bandit - target| > |guard - target|

bandit_reward = exp(-angle² / 2σ²)   if visible AND in_front else 0
guard_reward  = -bandit_reward
```

The Gaussian peaks when the Bandit lies on the Target → Guard line, on the side opposite the target (so light from the target to the guard would have to pass through the Bandit). The `visible` gate is the new piece versus Sun-Blocking — no reward when the guard can't see the target anyway.

## Implementation

- **Earth target ECEF** is precomputed once at `Game.__post_init__` via `astrojax.position_geodetic_to_ecef(lat, lon, alt)`. Cached on the `target_ecef_m` field (shape `(3,)`).
- **Per-step ECI**: `astrojax.rotation_ecef_to_eci(epoch)` rotates ECEF → ECI via GMST.
- **Vehicle ECI**: same RTN→ECI helper as Sun-Blocking (`games/_frames.py`).
- **Elevation**: angle between local-up-at-target and the (observer - target) direction.

## Knobs

| Knob | Default | Description |
|---|---|---|
| `target_lat_deg` | `37.4` | WGS84 latitude (degrees) of the Earth surface target |
| `target_lon_deg` | `-122.2` | WGS84 longitude (degrees) |
| `target_alt_m` | `0.0` | WGS84 altitude (meters above ellipsoid) |
| `min_elevation_deg` | `5.0` | Visibility threshold — reward gated to 0 below this |
| `angle_sigma_deg` | `5.0` | Gaussian width on the collinearity angle |

`target_ecef_m` is `init=False` and computed in `__post_init__`. It survives serialization round-trip — the user knobs (`lat`, `lon`, `alt`) round-trip through JSON, and the deserialization re-derives `target_ecef_m`.

## Termination

`MaxStepsOrBreach` with `breach_distance_m=0.0` — termination on `max_steps` only.

## Builder

```python
from orbital_game import make_observation_blocking

cfg = make_observation_blocking(
    target_lat_deg=51.5,        # London
    target_lon_deg=-0.1,
    min_elevation_deg=10.0,
    angle_sigma_deg=3.0,
    max_horizon_s=5400.0,
    dt=10.0,
    seed=0,
)
```

## Suggested experiments

- **Target sweep**: vary `target_lat_deg` from −60° to +60° to study how target latitude relative to orbit inclination affects observability windows and bandit positioning difficulty.
- **Elevation gate study**: vary `min_elevation_deg` from 5° (relaxed visibility) to 30° (strict). Higher thresholds mean shorter visibility windows → harder bandit problem.
- **Coordinated guards**: extend to `n_guards > 1` and test whether multiple Guards spread across the orbit can keep target observability above some threshold.

## Source

- Game subtype + reward + builder: [`src/orbital_game/games/observation_blocking.py`](https://github.com/duncaneddy/orbital-game/blob/main/src/orbital_game/games/observation_blocking.py)
- Tests: [`tests/test_game_observation_blocking.py`](https://github.com/duncaneddy/orbital-game/blob/main/tests/test_game_observation_blocking.py)
