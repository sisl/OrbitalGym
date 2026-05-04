# Sun-Blocking

## Quickstart

```python
import jax
from orbital_game import OrbitalGameEnv, BySide, make_sun_blocking
from orbital_game.policies import ZeroControl
from orbital_game.rollout import rollout

cfg = make_sun_blocking()
env = OrbitalGameEnv(cfg)
guard = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
bandit = ZeroControl(n_vehicles=cfg.n_bandits, action_dim=3)
traj = rollout(env, BySide(guard=guard, bandit=bandit),
               BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None),
               jax.random.PRNGKey(0), n_steps=cfg.max_steps)
```

The bandit is rewarded for occluding the guard's view of the Sun — interposing along the line from the bandit to the Sun, at a desired standoff distance. The reward is the KSP-DG SB1 formulation, signed so the *guard* is rewarded when the configuration is reversed.

## What the reward shapes

The reward at each timestep is

```
bandit_r = -û_BG · û_BS  *  exp(-decay * (d_BG - d_target)²)
guard_r  = -bandit_r
```

with vertex at the bandit. Three regimes:

- **Peak (`+1`)** — the bandit is between sun and guard at distance `target_viewing_distance_m`. û_BG points away from the sun, û_BS points toward the sun, so their dot product is ≈ -1, and `-dot ≈ +1`.
- **Trough (`-1`)** — the *guard* is between sun and bandit at the same distance. The dot product flips sign.
- **Zero** — far from either configuration, either because the geometry is perpendicular (dot product ≈ 0) or because `|d_BG - d_target|` is large (range factor crushes everything).

The game is zero-sum and bounded `[-1, +1]`. There is no separate angular σ — the dot product is its own peak-shape.

The reward looks up Sun ECI position via `astrojax.sun_position(epoch)` and converts vehicle RTN positions to ECI via the shared frame helper in `games/_frames.py`. The pure formula is exposed as `orbital_game.games.sun_blocking.sun_blocking_kernel(...)` and is used directly by the position-sweep diagnostic.

## Termination

`max_steps` only. Sun-Blocking has no capture concept.

## Builder

```python
from orbital_game import make_sun_blocking

cfg = make_sun_blocking(
    target_viewing_distance_m=500.0,
    range_decay_coef=4.0e-6,
    max_horizon_s=5400.0,    # ~90 min, one full LEO orbit
    seed=0,
)
```

The default horizon is one full LEO orbit so the Sun-guard-bandit geometry varies meaningfully across the episode.

## Knobs at a glance

| Field | Type | Default | What it does |
|---|---|---|---|
| `n_guards` | `int` | `1` | Observer count. |
| `n_bandits` | `int` | `1` | Blocker count. |
| `target_viewing_distance_m` | `float` | `500.0` | Desired bandit-guard standoff at the reward peak. |
| `range_decay_coef` | `float` | `4.0e-6` | Gaussian decay coefficient (1/m²); lower → wider peak. |
| `max_horizon_s` | `float` | `5400.0` | Total duration (one LEO orbit). |
| `dt` | `float` | `10.0` | Step size. |
| `seed` | `int` | `0` | PRNG seed. |

## Suggested experiments

- **Bandit positioning.** Train a bandit policy to maximize cumulative reward — i.e., maintain interposition near `target_viewing_distance_m` over the orbit.
- **Guard counter-positioning.** Train a guard to actively maneuver to push the bandit out of the peak region (or into the trough).
- **Range-peak sweep.** Vary `target_viewing_distance_m` from 100 m to 2000 m. The reward peak sits at that range; the IC sampler scale should scale alongside or the bandit will rarely visit the peak in default rollouts.

## Sanity-check notebook

[`examples/games/sun_blocking.ipynb`](https://github.com/sisl/orbital-game/blob/main/examples/games/sun_blocking.ipynb) is a full walkthrough that builds a SB scenario, runs a rollout, plots the rollout-time diagnostic, and renders the 2D position-sweep surface — the latter is the canonical reward-shape verification, and it should match the [KSP-DG SB1](https://github.com/mit-ll/spacegym-kspdg/blob/main/src/kspdg/sb1/sb1_base.py) reference shape.

## Where to next

- **API:** [`SunBlocking`](../api/games.md).
- **In depth:** [Symmetric core](../in-depth/symmetric-core.md), [Observations](../in-depth/observations.md).
