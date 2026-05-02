# Pursuit-Evasion

## Quickstart

```python
import jax
from orbital_game import OrbitalGameEnv, BySide, make_pursuit_evasion
from orbital_game.policies.library import ZeroControl
from orbital_game.rollout import rollout

cfg = make_pursuit_evasion()
env = OrbitalGameEnv(cfg)
guard = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
bandit = ZeroControl(n_vehicles=cfg.n_bandits, action_dim=3)
traj = rollout(env, BySide(guard=guard, bandit=bandit),
               BySide(guard=lambda c, s, k: None, bandit=lambda c, s, k: None),
               jax.random.PRNGKey(0), n_steps=cfg.max_steps)
```

For the active-pursuer walkthrough see [T3](../tutorials/t3-active-pursuer.md).

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

## Knobs at a glance

| Field | Type | Default | What it does |
|---|---|---|---|
| `n_guards` | `int` | `1` | Evader count. |
| `n_bandits` | `int` | `1` | Pursuer count. |
| `capture_distance_m` | `float` | `10.0` | Episode terminates on relative distance below this. |
| `max_horizon_s` | `float` | `2000.0` | Total duration. |
| `dt` | `float` | `10.0` | Step size. |
| `seed` | `int` | `0` | PRNG seed. |

## Suggested experiments

- **Optimal evasion.** Train a guard policy under a fixed bandit pursuer (e.g. proportional navigation).
- **Self-play.** Train both sides simultaneously via the [PettingZoo adapter](../adapters/pettingzoo.md) and observe equilibrium-ish capture rates.
- **Asymmetric capabilities.** Vary `guard_params.max_thrust_n` against `bandit_params.max_thrust_n` to study how the thrust ratio affects capture probability.

## Where to next

- **Tutorials:** [T2 — RTN scenario](../tutorials/t2-rtn-scenario.md), [T3 — Active pursuer](../tutorials/t3-active-pursuer.md), [T4 — Short-horizon search](../tutorials/t4-short-horizon-search.md).
- **API:** [`PursuitEvasion`](../api/games.md).
- **In depth:** [Symmetric core](../in-depth/symmetric-core.md), [Observations](../in-depth/observations.md), [Belief](../in-depth/belief.md).
