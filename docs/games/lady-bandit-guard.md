# Lady-Bandit-Guard

## Quickstart

```python
import jax
from orbital_game import OrbitalGameEnv, SingleAgentView, make_lady_bandit_guard
from orbital_game.policies.library import ZeroControl
from orbital_game.rollout import rollout_single_agent

cfg = make_lady_bandit_guard()
view = SingleAgentView(OrbitalGameEnv(cfg))
guard = ZeroControl(n_vehicles=cfg.n_guards, action_dim=3)
traj = rollout_single_agent(view, guard, lambda c, s, k: None,
                            jax.random.PRNGKey(0), n_steps=cfg.max_steps)
```

For a full walkthrough see [T1 — First rollout](../tutorials/t1-first-rollout.md).

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

## Knobs at a glance

| Field | Type | Default | What it does |
|---|---|---|---|
| `n_guards` | `int` | `1` | Number of guard vehicles. |
| `n_bandits` | `int` | `1` | Number of bandit vehicles. |
| `breach_distance_m` | `float` | `10.0` | Episode terminates when any guard is within this radius of the reference origin. |
| `max_horizon_s` | `float` | `2000.0` | Total episode duration in seconds. |
| `dt` | `float` | `10.0` | Step size in seconds. |
| `seed` | `int` | `0` | PRNG seed for IC sampling. |

For the full field list see [API → `LadyBanditGuard`](../api/games.md).

## Suggested experiments

- **Sanity baseline.** Run with zero control on both sides — no breach occurs, the episode ends at `max_steps`. This is what `examples/reference_scenario.py` does.
- **Bandit attack.** Replace `cfg.bandit_scripted_policy` with a heuristic that maneuvers toward the reference origin; the zero-control guard should eventually lose.
- **Guard station-keeping.** Train a guard policy to minimize the negative reward (stay close to the reference orbit) under bandit perturbations.

## Sanity-check notebook

[`examples/games/lady_bandit_guard.ipynb`](https://github.com/sisl/orbital-game/blob/main/examples/games/lady_bandit_guard.ipynb) is a full walkthrough that builds an LBG scenario, runs a rollout, and renders the rollout diagnostic plus the 2D guard-position reward surface.

## Where to next

- **Tutorial:** [T1 — First rollout](../tutorials/t1-first-rollout.md).
- **API:** [`LadyBanditGuard`](../api/games.md).
- **In depth:** [Symmetric core](../in-depth/symmetric-core.md), [State layout](../in-depth/state-layout.md).
