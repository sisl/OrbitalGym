# Observation-Blocking

The bandit blocks the guard's view of an Earth surface target — but only when the target is observable in the first place. This models a spaceborne ISR-denial scenario: the guard satellite is tasked with observing a fixed lat/lon target, and the bandit positions itself to occlude that line of sight whenever the target is actually visible from the guard.

## What the reward shapes

Three conditions are checked each step:

1. The target is **visible** from the guard — the elevation angle from target to guard exceeds `min_elevation_deg`. Below that threshold, the target is below the horizon and reward is zero regardless of geometry.
2. The bandit is **collinear** with the target-to-guard line, scored as a Gaussian on the angle.
3. The bandit is **in front** of the guard from the target's perspective — closer to the bandit than to the target itself, so light from the target to the guard would have to pass through it.

When all three hold, the bandit gets `exp(-angle² / 2σ²)`; otherwise zero. The game is zero-sum.

The visibility gate is the meaningful difference from Sun-Blocking — there is no point rewarding occlusion of a target the guard could not see anyway.

## Implementation notes

The Earth target's ECEF position is precomputed once in `Game.__post_init__` from `(target_lat_deg, target_lon_deg, target_alt_m)` via `astrojax.position_geodetic_to_ecef`. The `target_ecef_m` field is `init=False` and survives JSON round-trip — only the user-provided lat/lon/alt knobs are serialized; the ECEF cache is rederived on load.

Per step, ECEF is rotated to ECI via GMST, vehicle RTN is converted to ECI via the same frame helper as Sun-Blocking, and the elevation angle is computed between local-up at the target and the target-to-observer direction.

## Termination

`max_steps` only.

## Builder

```python
from orbital_game import make_observation_blocking

cfg = make_observation_blocking(
    target_lat_deg=51.5,        # London
    target_lon_deg=-0.1,
    min_elevation_deg=10.0,
    angle_sigma_deg=3.0,
    max_horizon_s=5400.0,
    seed=0,
)
```

For knob details, see `ObservationBlocking` in the [API reference](../api/games.md).

## Suggested experiments

- **Target latitude sweep.** Vary `target_lat_deg` from −60° to +60° to study how target latitude relative to orbit inclination affects observability windows and bandit difficulty.
- **Elevation gate study.** Vary `min_elevation_deg` from 5° (relaxed) to 30° (strict). Higher thresholds shorten visibility windows and harden the bandit problem.
- **Coordinated guards.** Extend to `n_guards > 1` and test whether multiple guards spread along the orbit can keep target observability above some threshold.
