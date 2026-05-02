# Lady-Bandit-Guard

The guard protects the reference orbit — the "Lady", currently virtual — from the bandit. This is a station-keeping / protection scenario: the guard's job is to stay near a protected asset while the bandit threatens to breach it.

## What the reward shapes

The guard's per-step reward is the negative sum of guard radial distances from the reference orbit origin (RTN frame). This pulls the guard toward the protected anchor; staying close earns less negative reward, drifting away costs more. The bandit's reward is fixed at zero in this game — Lady-Bandit-Guard is currently asymmetric, with the bandit acting as a scripted threat rather than an optimizing adversary. Future variants may promote the Lady to a third controllable agent or wire a zero-sum bandit reward.

## Termination

The episode ends on `max_steps` exhaustion or when any guard comes within `breach_distance_m` of the reference-orbit origin (the breach condition).

## Builder

```python
from orbital_game import make_lady_bandit_guard

cfg = make_lady_bandit_guard(
    breach_distance_m=10.0,
    max_horizon_s=2000.0,
    seed=0,
)
```

Defaults match the reference scenario: a 1 km radial-ellipse co-orbit with the guard at phase 0 and the bandit at phase π, 200 steps at 10 s each.

For knob details, see `LadyBanditGuard` in the [API reference](../api/games.md).

## Suggested experiments

- **Sanity baseline.** Run with zero control on both sides — no breach occurs, the episode ends at `max_steps`. This is what `examples/reference_scenario.py` does.
- **Bandit attack.** Replace `cfg.bandit_scripted_policy` with a heuristic that maneuvers toward the reference origin; the zero-control guard should eventually lose.
- **Guard station-keeping.** Train a guard policy to minimize the negative reward (stay close to the reference orbit) under bandit perturbations.
