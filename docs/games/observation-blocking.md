# Observation-Blocking

## Quickstart

```python
import jax
from orbital_game import OrbitalGameEnv, BySide, make_observation_blocking
from orbital_game.policies import ZeroControl
from orbital_game.rollout import rollout

cfg = make_observation_blocking()
env = OrbitalGameEnv(cfg)
guard = ZeroControl(n_vehicles=cfg.n_guards, command_cls=env.guard_command_cls)
bandit = ZeroControl(n_vehicles=cfg.n_bandits, command_cls=env.bandit_command_cls)
traj = rollout(env, BySide(guard=guard, bandit=bandit),
               BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None),
               jax.random.PRNGKey(0), n_steps=cfg.max_steps)
```

The bandit blocks the guard's view of an Earth surface target — but only when the target is observable in the first place. This models a spaceborne ISR-denial scenario: the guard satellite is tasked with observing a fixed lat/lon target, and the bandit positions itself to occlude that line of sight whenever the target is actually visible from the guard.

## What the reward shapes

The reward at each timestep is the KSP-DG-aligned shape with a visibility gate:

```
bandit_r = (-û_BG · û_BT) * exp(-decay * (d_BG - d_target)²) * 1[visible]
guard_r  = -bandit_r
```

where û_BT is the bandit-to-target unit vector. Same three regimes as Sun-Blocking (peak / trough / zero), multiplied by a hard visibility gate: if the target's elevation as seen from the guard is below `min_elevation_deg`, both sides' rewards are zero regardless of bandit geometry.

The game is zero-sum and bounded `[-1, +1]`. The visibility gate applies to *both* lobes — the guard "wins" (gets +reward) only when the target is visible *and* the bandit is in the wrong place.

## Implementation notes

The Earth target's ECEF position is precomputed once in `Game.__post_init__` from `(target_lat_deg, target_lon_deg, target_alt_m)` via `astrojax.position_geodetic_to_ecef`. Per step, ECEF is rotated to ECI via GMST (`astrojax.zero_eop()` — no EOP corrections), vehicle RTN is converted to ECI via the same frame helper as Sun-Blocking, and the elevation angle is computed between geocentric "up" at the target and the target-to-observer direction.

The pure formula is exposed as `orbital_game.games.observation_blocking.observation_blocking_kernel(...)` and the visibility gate as `target_visible_from_guard(...)`. Both are used directly by the position-sweep diagnostic.

## Termination

`max_steps` only.

## Builder

```python
from orbital_game import make_observation_blocking

cfg = make_observation_blocking(
    target_lat_deg=51.5,        # London
    target_lon_deg=-0.1,
    min_elevation_deg=10.0,
    target_viewing_distance_m=500.0,
    range_decay_coef=4.0e-6,
    max_horizon_s=5400.0,
    seed=0,
)
```

## Knobs at a glance

| Field | Type | Default | What it does |
|---|---|---|---|
| `n_guards` | `int` | `1` | Observer count. |
| `n_bandits` | `int` | `1` | Blocker count. |
| `target_lat_deg` | `float` | `37.4` | Target latitude. |
| `target_lon_deg` | `float` | `-122.2` | Target longitude. |
| `target_alt_m` | `float` | `0.0` | Target altitude. |
| `min_elevation_deg` | `float` | `5.0` | Visibility gate threshold. |
| `target_viewing_distance_m` | `float` | `500.0` | Desired bandit-guard standoff at the reward peak. |
| `range_decay_coef` | `float` | `4.0e-6` | Gaussian decay coefficient (1/m²). |
| `max_horizon_s` | `float` | `5400.0` | Total duration. |
| `dt` | `float` | `10.0` | Step size. |
| `seed` | `int` | `0` | PRNG seed. |

## Suggested experiments

- **Target latitude sweep.** Vary `target_lat_deg` from −60° to +60° to study how target latitude relative to orbit inclination affects observability windows and bandit difficulty.
- **Elevation gate study.** Vary `min_elevation_deg` from 5° (relaxed) to 30° (strict). Higher thresholds shorten visibility windows and harden the bandit problem.
- **Coordinated guards.** Extend to `n_guards > 1` and test whether multiple guards spread along the orbit can keep target observability above some threshold.

## Variants

The four customization axes (reward, termination, IC, observation) all
swap by `dataclasses.replace`. Two OB-flavored recipes:

### Variant 1: chasing bandit

Drop in `LeadInterceptPursuer` as the bandit. The bandit closes on the
guard rather than tracking the bandit-to-target line — a baseline
adversary that doesn't yet exploit the visibility gate, useful for
sanity-checking that an OB-aware bandit policy you're training actually
beats this naive chaser.

```python
--8<-- "tests/docs/test_games_observation_blocking_variants.py:variant-lead-intercept-bandit"
```

### Variant 2: tighter IC for evaluation

Shrink `sigma_radial_ellipse_m` so evaluation rollouts draw from a
narrow band around the canonical phase-0/phase-π/2 IC. Lower IC
variance reduces episode-return variance, which matters more here than
in PE because OB returns are gated by visibility windows.

```python
--8<-- "tests/docs/test_games_observation_blocking_variants.py:variant-tighter-ic"
```

For more axes, see [Extending → Customize rewards](../extending/customize-rewards.md),
[…termination](../extending/customize-termination.md),
[…IC sampling](../extending/customize-ic-sampling.md), and
[…observations](../extending/customize-observations.md).

## Built-in policies and adversaries

Sensible gallery picks for Observation-Blocking:

- **Heuristic policies (opponent):** [`LeadInterceptPursuer`](../extending/gallery.md#leadinterceptpursuer),
  [`JitteredPolicy`](../extending/gallery.md#jitteredpolicy) (wrap any of the above).
- **Controlled side:** any class from [Controlled-side cookbook](../extending/controlled-policy-cookbook.md)
  ([`HeuristicWithFallbackPolicy`](../extending/gallery.md#heuristicwithfallbackpolicy),
  [`CompositeActionPolicy`](../extending/gallery.md#compositeactionpolicy),
  [`MCTSPolicy`](../extending/gallery.md#mctspolicy)).

## Sanity-check notebook

[`examples/games/observation_blocking.ipynb`](https://github.com/sisl/orbital-game/blob/main/examples/games/observation_blocking.ipynb) is a full walkthrough that builds an OB scenario, runs a rollout, plots the rollout-time diagnostic (with shaded invisible windows), and renders the 2D position-sweep surface.

## Where to next

- **API:** [`ObservationBlocking`](../api/games.md).
- **In depth:** [Symmetric core](../in-depth/symmetric-core.md), [Observations](../in-depth/observations.md).
