# Pursuit-Evasion

A 1v1 zero-sum game. The bandit pursues, the guard evades. The reference orbit is a dynamical anchor only — it has no protected-asset semantics here.

## What the reward shapes

Reward is the relative distance between guard and bandit, with opposite sign on each side: the bandit is rewarded for closing distance, the guard for opening it. Because both vehicles share the RTN frame anchored on the reference orbit, the relative distance is just the difference of their RTN positions.

Zero-sum is a hard constraint: `guard_reward + bandit_reward == 0` at every step, and the test suite asserts this.

## Termination

Either `max_steps` exhaustion, or **capture** when the relative distance drops below `capture_distance_m`.

## Builder

```python
from orbital_game import make_pursuit_evasion

cfg = make_pursuit_evasion(
    capture_distance_m=10.0,
    max_horizon_s=2000.0,
    seed=0,
)
```

Defaults place the guard on a 1 km radial ellipse at phase 0 and the bandit on a smaller (500 m) ellipse with cross-track and along-track offset at phase π/2 — neither side is trivially captured nor trivially safe at IC.

For knob details, see `PursuitEvasion` in the [API reference](../api/games.md).

## Suggested experiments

- **Optimal evasion.** Train a guard policy under a fixed bandit pursuer (e.g. proportional navigation).
- **Self-play.** Train both sides simultaneously via the [PettingZoo adapter](../adapters/pettingzoo.md) and observe equilibrium-ish capture rates.
- **Asymmetric capabilities.** Vary `guard_params.max_thrust_n` against `bandit_params.max_thrust_n` to study how the thrust ratio affects capture probability.
