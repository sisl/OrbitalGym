# orbital-game

A JAX-native framework for two-side decision-making in orbital
scenarios. Same symmetric core drives single-agent (Gymnasium-shaped),
multi-agent (PettingZoo-shaped), and POMDP-shaped views of the same
underlying game. Four scenarios ship in the box.

## Choose your path

<div class="grid cards" markdown>

- :material-school: **New here**
  Start with [Tutorials](tutorials/index.md). Five guided walkthroughs
  from `pip install` to a working short-horizon planner running on GPU.

- :material-book-search: **Looking up an API**
  Jump to the [API reference](api/index.md). Auto-generated from
  in-code docstrings.

- :material-target: **Building a scenario**
  See [Game guides](games/index.md) for what each game rewards, then
  the [How-to recipes](how-to/index.md) for the task-by-task steps.

- :material-cog: **Understanding the design**
  Read [In depth](in-depth/index.md). Eleven deep-dives on the
  symmetric core, state layout, dynamics, observations, belief, and
  more.

</div>

## What's in the box

- **Four games:** [Lady-Bandit-Guard](games/lady-bandit-guard.md),
  [Pursuit-Evasion](games/pursuit-evasion.md),
  [Sun-Blocking](games/sun-blocking.md),
  [Observation-Blocking](games/observation-blocking.md).
- **Three adapters:** Gymnasium (single-agent RL), PettingZoo
  (multi-agent RL), and a POMDPPlanners-shape duck-typed protocol
  (belief-space planners).
- **Four observation channels:** full state, onboard GPS, range-limited,
  composite.
- **Linear and extended Kalman belief updaters**, plus a `BeliefRollout`
  helper.

## How fast

A whole episode compiles to one `jax.lax.scan`, so a single rollout is
fast. The bigger win is `vmap` over seeds: 1024 parallel rollouts on a
single GPU run in roughly the wall time of one. See
[T5 — GPU / MPS](tutorials/t5-acceleration.md).

## What's not here

- **No production planners.** [T4](tutorials/t4-short-horizon-search.md)
  builds a teaching-quality short-horizon search planner; for serious
  work plug into `mctx` or `pomdp-py`.
- **No J2 / drag dynamics yet.** HCW only. Adding a perturbed dynamics
  module is sketched in [In depth → Dynamics](in-depth/dynamics.md).
- **No attitude control.** Vehicles are point masses with impulsive
  thrust.

These are extension targets, not gaps in the design — the
component-and-registry architecture is built for adding them. See
[In depth](in-depth/index.md) for the patterns.
