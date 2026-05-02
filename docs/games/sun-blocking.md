# Sun-Blocking

## Quickstart

```python
import jax
from orbital_game import OrbitalGameEnv, BySide, make_sun_blocking
from orbital_game.policies.library import ZeroControl
from orbital_game.rollout import rollout

cfg = make_sun_blocking()
env = OrbitalGameEnv(cfg)
guard = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
bandit = ZeroControl(n_vehicles=cfg.n_bandits, action_dim=3)
traj = rollout(env, BySide(guard=guard, bandit=bandit),
               BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None),
               jax.random.PRNGKey(0), n_steps=cfg.max_steps)
```

The bandit is rewarded for occluding the guard's view of the Sun — forming an approximately collinear `Sun → guard → bandit` line with the bandit on the far-from-Sun side of the guard. The scenario models adversarial dazzling and sun-shadowing in proximity operations.

## What the reward shapes

Two geometric conditions must hold simultaneously:

1. The angle between the Sun-to-guard vector and the Sun-to-bandit vector is small (the three bodies are approximately collinear).
2. The bandit is farther from the Sun than the guard (it blocks light, rather than being blocked by the guard).

The reward is a Gaussian on the collinearity angle, gated to zero whenever the second condition fails. Smaller `angle_sigma_deg` makes the peak sharper — the bandit must align more precisely to score. The game is zero-sum: the guard's reward is the negative of the bandit's.

The reward looks up Sun ECI position via `astrojax.sun_position(epoch)` and converts vehicle RTN positions to ECI via the shared frame helper in `games/_frames.py`.

## Termination

`max_steps` only. Sun-Blocking has no capture concept.

## Builder

```python
from orbital_game import make_sun_blocking

cfg = make_sun_blocking(
    angle_sigma_deg=5.0,
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
| `angle_sigma_deg` | `float` | `5.0` | Gaussian sharpness on collinearity angle. |
| `max_horizon_s` | `float` | `5400.0` | Total duration (one LEO orbit). |
| `dt` | `float` | `10.0` | Step size. |
| `seed` | `int` | `0` | PRNG seed. |

## Suggested experiments

- **Bandit positioning.** Train a bandit policy to maximize cumulative reward — i.e., maximize sun-occlusion duration over the orbit.
- **Guard counter-positioning.** Train a guard to actively maneuver away from the bandit's occlusion line.
- **σ sweep.** Vary `angle_sigma_deg` from 1° to 30°. Smaller σ gives a harder, sharper-peaked problem; larger σ smooths the reward landscape.

## Where to next

- **API:** [`SunBlocking`](../api/games.md).
- **In depth:** [Symmetric core](../in-depth/symmetric-core.md), [Observations](../in-depth/observations.md).
